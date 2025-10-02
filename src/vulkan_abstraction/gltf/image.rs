pub struct Image {
    pub format: gltf::image::Format,
    pub height: usize,
    pub width: usize,
    pub raw_data: Vec<u8>,
}
