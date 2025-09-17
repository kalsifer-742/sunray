struct Camera {}

fn make_view_inverse_matrix() -> nalgebra::Matrix4<f32> {
    let eye = nalgebra::geometry::Point3::new(0.0, 0.0, 3.0);
    let target = nalgebra::geometry::Point3::new(0.0, 0.0, 0.0);
    let up = nalgebra::Vector3::new(0.0, -1.0, 0.0);
    //apparently vulkan uses right-handed coordinates
    let view = nalgebra::Isometry3::look_at_rh(&eye, &target, &up);

    let view_matrix: nalgebra::Matrix4<f32> = view.to_homogeneous();

    view_matrix.try_inverse().unwrap()
}

fn make_proj_inverse_matrix(dimensions: (u32, u32)) -> nalgebra::Matrix4<f32> {
    let proj = nalgebra::geometry::Perspective3::new(
        dimensions.0 as f32 / dimensions.1 as f32,
        3.14 / 2.0,
        0.1,
        1000.0,
    );

    let proj = proj.to_homogeneous();

    proj.try_inverse().unwrap()
}

fn update_camera() {
    let mem = uniform_buffer.map::<UniformBufferContents>()?;
    mem[0].proj_inverse =
        make_proj_inverse_matrix((core.image_extent().width, core.image_extent().height));
    mem[0].view_inverse = make_view_inverse_matrix();

    let origin = mem[0].view_inverse * nalgebra::Vector4::new(0.0, 0.0, 0.0, 1.0);
    let target = mem[0].proj_inverse * nalgebra::Vector4::new(0.0, 0.0, 1.0, 1.0);
    let target_normalized = target.normalize();
    let direction = mem[0].view_inverse
        * nalgebra::Vector4::new(
            target_normalized.x,
            target_normalized.y,
            target_normalized.z,
            0.0,
        );

    let origin = origin.xyz();
    let direction = direction.xyz().normalize();

    let fmt_vec = |v: nalgebra::Vector3<f32>| format!("({}, {}, {})", v.x, v.y, v.z);
    println!(
        "for screen center, ray origin={}, direction={}",
        fmt_vec(origin),
        fmt_vec(direction)
    );

    uniform_buffer.unmap();
}
