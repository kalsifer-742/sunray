pub(crate) fn env_var_as_bool(name: &str) -> Option<bool> {
    match std::env::var(name) {
        Ok(s) => match s.parse::<i32>() {
            Ok(v) => Some(v != 0),
            Err(_) => None,
        },
        Err(_) => None,
    }
}

pub(crate) fn iterate_image_extent(w: u32, h: u32) -> impl Iterator<Item = (u32, u32)> {
    (0..w * h).map(move |i| (i % w, i / w))
}

pub(crate) fn tuple_to_extent2d((width, height): (u32, u32)) -> ash::vk::Extent2D {
    ash::vk::Extent2D { width, height }
}

pub(crate) fn tuple_to_extent3d(tuple: (u32, u32)) -> ash::vk::Extent3D {
    tuple_to_extent2d(tuple).into()
}
