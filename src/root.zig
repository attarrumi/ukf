//! By convention, root.zig is the root source file when making a library.
const std = @import("std");
const math = std.math;

pub const UKFAHRS = struct {
    // ===== State: [ q0 q1 q2 q3 | bgx bgy bgz ] =====
    q: [4]f32 = .{ 1.0, 0.0, 0.0, 0.0 },
    b: [3]f32 = .{ 0.0, 0.0, 0.0 }, // gyro bias (rad/s)

    // Covariance 7x7
    P: [7][7]f32 = blk: {
        var M: [7][7]f32 = undefined;
        for (0..7) |i| {
            for (0..7) |j| M[i][j] = 0.0;
            M[i][i] = if (i < 4) 1e-2 else 1e-3; // q a bit larger than bias
        }
        break :blk M;
    },

    // Process noise Q 7x7 (quaternion small, bias random walk)
    Q: [7][7]f32 = blk: {
        var M: [7][7]f32 = undefined;
        for (0..7) |i| {
            for (0..7) |j| M[i][j] = 0.0;
        }
        // quaternion process noise
        for (0..4) |i| M[i][i] = 0.005; // dari 1e-6 -> 1e-5
        // bias random walk
        M[4][4] = 1e-6;
        M[5][5] = 1e-6;
        M[6][6] = 1e-6;
        break :blk M;
    },

    // Measurement noise R for [ax ay az mxh myh] (acc, mag horizontal), normalized
    R5: [5][5]f32 = [5][5]f32{
        [5]f32{ 0.15, 0, 0, 0, 0 },
        [5]f32{ 0, 0.15, 0, 0, 0 },
        [5]f32{ 0, 0, 0.15, 0, 0 },
        [5]f32{ 0, 0, 0, 0.3, 0 },
        [5]f32{ 0, 0, 0, 0, 0.3 },
    },

    is_initialized: bool = false,

    // ===== UKF parameters =====
    alpha: f32 = 0.3,
    kappa: f32 = 0.0,
    beta: f32 = 2.0,
    lambda: f32 = undefined,
    n: usize = 7,
    num_sig: usize = 15, // 2n+1

    Wm: [15]f32 = undefined,
    Wc: [15]f32 = undefined,
    // Sigma points in state space (7D)
    sigmas: [15][7]f32 = undefined,

    pub fn init() UKFAHRS {
        var f = UKFAHRS{};
        f.lambda = f.alpha * f.alpha * (@as(f32, @floatFromInt(f.n)) + f.kappa) - @as(f32, @floatFromInt(f.n));
        const n_plus_lambda = @as(f32, @floatFromInt(f.n)) + f.lambda;

        // Weights
        f.Wm[0] = f.lambda / n_plus_lambda;
        f.Wc[0] = f.Wm[0] + (1.0 - f.alpha * f.alpha + f.beta);
        const cw = 1.0 / (2.0 * n_plus_lambda);
        for (1..f.num_sig) |i| {
            f.Wm[i] = cw;
            f.Wc[i] = cw;
        }
        return f;
    }

    // ======== PREDICT ========
    pub fn predict(self: *UKFAHRS, gx: f32, gy: f32, gz: f32, dt: f32) void {
        // 1) P_total = P + Q (+ reg)
        var P_total: [7][7]f32 = undefined;
        for (0..7) |i| {
            for (0..7) |j| P_total[i][j] = self.P[i][j] + self.Q[i][j];
            P_total[i][i] += 1e-9;
        }

        // 2) A = (n+lambda) * P_total
        const scale = @as(f32, @floatFromInt(self.n)) + self.lambda;
        var A: [7][7]f32 = undefined;
        for (0..7) |i| {
            for (0..7) |j| A[i][j] = scale * P_total[i][j];
        }

        // 3) Cholesky (lower) 7x7
        var L: [7][7]f32 = zero77();
        choleskyLower(&A, &L);

        // 4) Generate sigma points around mean x = [q,b]
        const x: [7]f32 = .{ self.q[0], self.q[1], self.q[2], self.q[3], self.b[0], self.b[1], self.b[2] };
        self.sigmas[0] = x;
        for (0..7) |i| {
            var xp: [7]f32 = x;
            var xm: [7]f32 = x;
            for (0..7) |r| {
                xp[r] += L[r][i];
                xm[r] -= L[r][i];
            }
            self.sigmas[1 + i] = xp;
            self.sigmas[1 + 7 + i] = xm;
        }

        // 5) Propagate each sigma through dynamics: q̇ = 0.5*Omega(ω-b)*q ; ḃ = 0
        for (0..self.num_sig) |i| {
            var s = self.sigmas[i];
            var q = s[0..4].*;
            const bx = s[4];
            const by = s[5];
            const bz = s[6];

            const wx = gx - bx;
            const wy = gy - by;
            const wz = gz - bz;

            // dq/dt
            const q0 = q[0];
            const q1 = q[1];
            const q2 = q[2];
            const q3 = q[3];

            const dq0 = 0.5 * (-q1 * wx - q2 * wy - q3 * wz);
            const dq1 = 0.5 * (q0 * wx + q2 * wz - q3 * wy);
            const dq2 = 0.5 * (q0 * wy - q1 * wz + q3 * wx);
            const dq3 = 0.5 * (q0 * wz + q1 * wy - q2 * wx);

            q[0] = q0 + dq0 * dt;
            q[1] = q1 + dq1 * dt;
            q[2] = q2 + dq2 * dt;
            q[3] = q3 + dq3 * dt;

            normalize4(&q);

            // Bias random-walk di model: di sini biarkan tetap (diserap Q)
            s[0] = q[0];
            s[1] = q[1];
            s[2] = q[2];
            s[3] = q[3];
            // s[4..6] tetap

            self.sigmas[i] = s;
        }

        // 6) Mean state (quaternion mean + bias mean)
        var q_sigmas: [15][4]f32 = undefined;
        for (0..self.num_sig) |i| {
            q_sigmas[i] = .{ self.sigmas[i][0], self.sigmas[i][1], self.sigmas[i][2], self.sigmas[i][3] };
        }
        const q_mean = quaternionMean(q_sigmas, self.Wm);

        var b_mean: [3]f32 = .{ 0.0, 0.0, 0.0 };
        for (0..self.num_sig) |i| {
            const w = self.Wm[i];
            b_mean[0] += w * self.sigmas[i][4];
            b_mean[1] += w * self.sigmas[i][5];
            b_mean[2] += w * self.sigmas[i][6];
        }

        self.q = q_mean;
        self.b = b_mean;

        // 7) Predicted covariance
        var P_pred: [7][7]f32 = zero77();
        for (0..self.num_sig) |i| {
            const w = self.Wc[i];
            const dx: [7]f32 = .{
                self.sigmas[i][0] - self.q[0],
                self.sigmas[i][1] - self.q[1],
                self.sigmas[i][2] - self.q[2],
                self.sigmas[i][3] - self.q[3],
                self.sigmas[i][4] - self.b[0],
                self.sigmas[i][5] - self.b[1],
                self.sigmas[i][6] - self.b[2],
            };
            for (0..7) |r| {
                for (0..7) |c| P_pred[r][c] += w * dx[r] * dx[c];
            }
        }
        self.P = P_pred;
        symmetrize77(&self.P);
        for (0..7) |i| self.P[i][i] += 1e-12;
    }

    // ======== UPDATE (9DOF) ========
    pub fn update9DOF(self: *UKFAHRS, ax: f32, ay: f32, az: f32, mx: f32, my: f32, mz: f32) void {
        // Normalize accel & mag
        const accn = normalize3(ax, ay, az) orelse return;
        const magn = normalize3(mx, my, mz) orelse return;
        const axn = accn[0];
        const ayn = accn[1];
        const azn = accn[2];
        var mxn = magn[0];
        var myn = magn[1];
        const mzn = magn[2];

        // Tilt compensation for mag (pakai accel)
        const roll = math.atan2(ayn, azn);
        const pitch = math.atan2(-axn, math.sqrt(ayn * ayn + azn * azn));
        const cr = @cos(roll);
        const sr = @sin(roll);
        const cp = @cos(pitch);
        const sp = @sin(pitch);
        const mxh = mxn * cp + mzn * sp;
        const myh = mxn * sr * sp + myn * cr - mzn * sr * cp;
        mxn = mxh;
        myn = myh;

        // Init from accel jika belum
        if (!self.is_initialized) {
            self.initializeFromAccel(axn, ayn, azn);
            self.is_initialized = true;
        }

        // 1) Transform sigma ke ruang ukur z = [g_body, m_horizontal]
        var Z: [15][5]f32 = undefined;
        for (0..self.num_sig) |i| {
            const q = .{ self.sigmas[i][0], self.sigmas[i][1], self.sigmas[i][2], self.sigmas[i][3] };

            // Gravity prediction (body frame) via quaternion
            const hx0 = 2.0 * (q[1] * q[3] - q[0] * q[2]); // x
            const hx1 = 2.0 * (q[0] * q[1] + q[2] * q[3]); // y
            const hx2 = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]; // z

            // Magnetic heading components (assuming reference field in world ≈ [1,0,0])
            const mx_pred = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]);
            const my_pred = 2.0 * (q[1] * q[2] + q[0] * q[3]);

            Z[i] = .{ hx0, hx1, hx2, mx_pred, my_pred };
        }

        // 2) z mean
        var z_pred: [5]f32 = .{ 0, 0, 0, 0, 0 };
        for (0..self.num_sig) |i| {
            const w = self.Wm[i];
            for (0..5) |j| z_pred[j] += w * Z[i][j];
        }

        // 3) Innovation covariance S
        var S: [5][5]f32 = self.R5; // start with R
        for (0..5) |i| S[i][i] += 1e-9;
        for (0..self.num_sig) |i| {
            const w = self.Wc[i];
            var dz: [5]f32 = undefined;
            for (0..5) |j| dz[j] = Z[i][j] - z_pred[j];
            for (0..5) |r| {
                for (0..5) |c| S[r][c] += w * dz[r] * dz[c];
            }
        }

        // 4) Cross-covariance P_xz (7x5)
        var P_xz: [7][5]f32 = zero75();
        for (0..self.num_sig) |i| {
            const w = self.Wc[i];
            const dx: [7]f32 = .{
                self.sigmas[i][0] - self.q[0],
                self.sigmas[i][1] - self.q[1],
                self.sigmas[i][2] - self.q[2],
                self.sigmas[i][3] - self.q[3],
                self.sigmas[i][4] - self.b[0],
                self.sigmas[i][5] - self.b[1],
                self.sigmas[i][6] - self.b[2],
            };
            var dz: [5]f32 = undefined;
            for (0..5) |j| dz[j] = Z[i][j] - z_pred[j];

            for (0..7) |r| {
                for (0..5) |c| P_xz[r][c] += w * dx[r] * dz[c];
            }
        }

        // 5) K = P_xz * inv(S)
        var invS: [5][5]f32 = undefined;
        if (!invert5x5(S, &invS)) {
            // tambah regularisasi dan coba lagi
            for (0..5) |i| S[i][i] += 1e-3;
            if (!invert5x5(S, &invS)) return;
        }
        var K: [7][5]f32 = undefined;
        for (0..7) |r| {
            for (0..5) |c| {
                var sum: f32 = 0.0;
                for (0..5) |k| sum += P_xz[r][k] * invS[k][c];
                K[r][c] = sum;
            }
        }

        // 6) Update state dengan pengukuran z_meas
        const z_meas = [5]f32{ axn, ayn, azn, mxn, myn };
        var innov: [5]f32 = undefined;
        for (0..5) |i| innov[i] = z_meas[i] - z_pred[i];

        var dx: [7]f32 = .{ 0, 0, 0, 0, 0, 0, 0 };
        for (0..7) |r| {
            for (0..5) |c| dx[r] += K[r][c] * innov[c];
        }

        // apply
        self.q[0] += dx[0];
        self.q[1] += dx[1];
        self.q[2] += dx[2];
        self.q[3] += dx[3];
        normalize4(&self.q);

        self.b[0] += dx[4];
        self.b[1] += dx[5];
        self.b[2] += dx[6];

        // 7) Update covariance: P = P - K S K^T, lalu simetrisasi + reg
        var KS: [7][5]f32 = undefined;
        for (0..7) |r| {
            for (0..5) |c| {
                var sum: f32 = 0.0;
                for (0..5) |k| sum += K[r][k] * S[k][c];
                KS[r][c] = sum;
            }
        }
        for (0..7) |r| {
            for (0..7) |c| {
                var sum: f32 = 0.0;
                for (0..5) |k| sum += KS[r][k] * K[c][k];
                self.P[r][c] -= sum;
            }
        }
        symmetrize77(&self.P);
        for (0..7) |i| self.P[i][i] += 1e-12;
    }

    // ===== UTIL =====
    fn initializeFromAccel(self: *UKFAHRS, ax: f32, ay: f32, az: f32) void {
        const roll = math.atan2(ay, az);
        const pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az));
        const yaw = 0.0;

        const cy = @cos(yaw * 0.5);
        const sy = @sin(yaw * 0.5);
        const cp = @cos(pitch * 0.5);
        const sp = @sin(pitch * 0.5);
        const cr = @cos(roll * 0.5);
        const sr = @sin(roll * 0.5);

        self.q[0] = cr * cp * cy + sr * sp * sy;
        self.q[1] = sr * cp * cy - cr * sp * sy;
        self.q[2] = cr * sp * cy + sr * cp * sy;
        self.q[3] = cr * cp * sy - sr * sp * cy;
        normalize4(&self.q);

        self.b = .{ 0.0, 0.0, 0.0 };
    }

    pub fn getEuler(self: *UKFAHRS, declination_deg: f32) struct {
        roll: f32,
        pitch: f32,
        yaw: f32,
    } {
        const q0 = self.q[0];
        const q1 = self.q[1];
        const q2 = self.q[2];
        const q3 = self.q[3];

        var roll = math.atan2(2.0 * (q0 * q1 + q2 * q3), 1.0 - 2.0 * (q1 * q1 + q2 * q2));
        // var pitch = math.asin(2.0 * (q0 * q2 - q1 * q3));
        // Pitch (y-axis rotation)

        var pitch: f32 = 0;
        const sinp: f32 = 2.0 * (q0 * q2 - q3 * q1);
        if (@abs(sinp) >= 1) {
            const pi: f32 = math.pi / 2.0;
            pitch = math.copysign(pi, sinp); // use 90 degrees if out of range
        } else {
            pitch = math.asin(sinp);
        }
        var yaw = math.atan2(2.0 * (q1 * q2 + q0 * q3), 1.0 - 2.0 * (q2 * q2 + q3 * q3));

        const rad2deg = 57.2958;
        roll *= rad2deg;
        pitch *= rad2deg;
        yaw *= rad2deg;

        // Declination diberikan dalam derajat → kurangi langsung dalam derajat
        yaw -= declination_deg;
        if (yaw > 180.0) yaw -= 360.0;
        if (yaw < -180.0) yaw += 360.0;

        return .{ .roll = roll, .pitch = pitch, .yaw = yaw };
    }
};

// ================== Helpers ==================

fn zero77() [7][7]f32 {
    var M: [7][7]f32 = undefined;
    for (0..7) |i| {
        for (0..7) |j| M[i][j] = 0.0;
    }
    return M;
}
fn zero75() [7][5]f32 {
    var M: [7][5]f32 = undefined;
    for (0..7) |i| {
        for (0..5) |j| M[i][j] = 0.0;
    }
    return M;
}

fn normalize4(q: *[4]f32) void {
    var n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (n < 1e-20) n = 1.0;
    const inv = 1.0 / n;
    for (0..4) |i| q.*[i] *= inv;
}

fn normalize3(x: f32, y: f32, z: f32) ?[3]f32 {
    const n = math.sqrt(x * x + y * y + z * z);
    if (n < 1e-12) return null;
    const inv = 1.0 / n;
    return .{ x * inv, y * inv, z * inv };
}

// Simple lower-triangular Cholesky with safety clamp
fn choleskyLower(A: *[7][7]f32, L: *[7][7]f32) void {
    for (0..7) |i| {
        for (0..7) |j| L.*[i][j] = 0.0;
    }
    for (0..7) |i| {
        for (0..i + 1) |j| {
            var sum: f32 = 0.0;
            for (0..j) |k| sum += L.*[i][k] * L.*[j][k];
            if (i == j) {
                const v = A.*[i][i] - sum;
                L.*[i][j] = math.sqrt(@max(v, 1e-18));
            } else {
                if (L.*[j][j] > 1e-18) {
                    L.*[i][j] = (A.*[i][j] - sum) / L.*[j][j];
                } else {
                    L.*[i][j] = 0.0;
                }
            }
        }
    }
}

// Symmetrize and keep PD-ish
fn symmetrize77(P: *[7][7]f32) void {
    for (0..7) |i| {
        for (0..7) |j| {
            const avg = 0.5 * (P.*[i][j] + P.*[j][i]);
            P.*[i][j] = avg;
            P.*[j][i] = avg;
        }
    }
}

fn quaternionMean(sigmas: [15][4]f32, weights: [15]f32) [4]f32 {
    var q_mean = sigmas[0];
    var it: usize = 0;
    while (it < 10) : (it += 1) {
        var sum: [4]f32 = .{ 0, 0, 0, 0 };
        for (0..15) |i| {
            const q = sigmas[i];
            const dot = q_mean[0] * q[0] + q_mean[1] * q[1] + q_mean[2] * q[2] + q_mean[3] * q[3];
            const sgn: f32 = if (dot >= 0.0) 1.0 else -1.0;
            for (0..4) |j| sum[j] += weights[i] * sgn * q[j];
        }
        const n = math.sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2] + sum[3] * sum[3]);
        if (n > 1e-20) {
            const inv = 1.0 / n;
            const new_q: [4]f32 = .{ sum[0] * inv, sum[1] * inv, sum[2] * inv, sum[3] * inv };
            // stop early if converged
            const d = @abs(new_q[0] - q_mean[0]) + @abs(new_q[1] - q_mean[1]) + @abs(new_q[2] - q_mean[2]) + @abs(new_q[3] - q_mean[3]);
            q_mean = new_q;
            if (d < 1e-6) break;
        } else {
            break;
        }
    }
    return q_mean;
}

// Invert 5x5 with Gauss-Jordan
fn invert5x5(a: [5][5]f32, out: *[5][5]f32) bool {
    var m: [5][10]f32 = undefined;
    for (0..5) |i| {
        for (0..5) |j| m[i][j] = a[i][j];
        for (0..5) |j| m[i][5 + j] = if (i == j) 1.0 else 0.0;
    }

    for (0..5) |col| {
        var piv = col;
        var maxv = @abs(m[col][col]);
        for (col + 1..5) |r| {
            const av = @abs(m[r][col]);
            if (av > maxv) {
                maxv = av;
                piv = r;
            }
        }
        if (maxv < 1e-12) return false;
        if (piv != col) {
            for (0..10) |c| {
                const tmp = m[col][c];
                m[col][c] = m[piv][c];
                m[piv][c] = tmp;
            }
        }
        const pivot_val = m[col][col];
        for (0..10) |c| m[col][c] /= pivot_val;

        for (0..5) |r| {
            if (r == col) continue;
            const factor = m[r][col];
            if (factor == 0.0) continue;
            for (0..10) |c| m[r][c] -= factor * m[col][c];
        }
    }

    for (0..5) |i| {
        for (0..5) |j| out[i][j] = m[i][5 + j];
    }
    return true;
}
