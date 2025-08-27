

  const ukf = b.dependency("ukf", .{
        .target = target,
        .optimize = optimize,
    });

    exe.root_module.addImport("ukf", ukf.module("ukf"));


    const UKFAHRS = @import("ukf").UKFAHRS;
    var filter = UKFAHRS.init();

    filter.predict(normalized.gx, normalized.gy, normalized.gz, deltat);
        filter.update9DOF(
            normalized.ax,
            normalized.ay,
            normalized.az,
            normalized.mx,
            normalized.my,
            normalized.mz,
        );

        const euler = filter.getEuler(0.0133808576);
        std.debug.print("Roll {d:.0}, Pitch {d:.0} Yaw {d:.0}\n", .{ euler.roll, euler.pitch, euler.yaw });
