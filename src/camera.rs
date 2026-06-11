use nalgebra as na;

#[derive(Clone, Copy)]
pub struct Camera {
    position: na::Point3<f32>,
    target: na::Point3<f32>,
    fov_y: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: na::point![0.0, 0.0, 1.0],
            target: na::point![0.0, 0.0, 0.0],
            fov_y: 45.0,
        }
    }
}
#[derive(Clone, Copy)]
#[repr(C, packed)]
pub(crate) struct CameraMatrices {
    pub view_inverse: na::Matrix4<f32>,
    pub proj_inverse: na::Matrix4<f32>,
    pub view_proj: na::Matrix4<f32>,
    pub prev_view_proj: na::Matrix4<f32>,
}

impl Camera {
    pub fn new(position: na::Point3<f32>, target: na::Point3<f32>, fov_y: f32) -> Self {
        Self { position, target, fov_y }
    }

    pub(crate) fn as_matrices(&self, extent: ash::vk::Extent3D) -> CameraMatrices {
        let eye = self.position;
        let target = self.target;
        let up = &na::vector![0.0, 1.0, 0.0];

        //view-space: camera pov
        let view = na::Isometry3::look_at_rh(&eye, &target, up);
        //clip_space: normalised coordinates adding perspective
        let projection = na::Perspective3::new(
            extent.width as f32 / extent.height as f32,
            self.fov_y.to_radians(),
            0.1,   //render everything after this distance
            100.0, //discard everything after this distance
        );

        let view_homogeneous = view.to_homogeneous();
        let mut proj_homogeneous = projection.to_homogeneous();

        proj_homogeneous[(1, 1)] *= -1.0;

        let view_inverse = view_homogeneous.try_inverse().unwrap();
        let proj_inverse = proj_homogeneous.try_inverse().unwrap();
        let view_proj = proj_homogeneous * view_homogeneous;

        CameraMatrices {
            view_inverse,
            proj_inverse,
            view_proj,
            prev_view_proj: nalgebra::zero(),
        }
    }

    pub fn position(&self) -> na::Point3<f32> {
        self.position
    }

    pub fn target(&self) -> na::Point3<f32> {
        self.target
    }

    pub fn fov_y(&self) -> f32 {
        self.fov_y
    }

    pub fn set_position(mut self, position: na::Point3<f32>) -> Self {
        self.position = position;

        self
    }

    pub fn set_target(mut self, target: na::Point3<f32>) -> Self {
        self.target = target;

        self
    }

    pub fn set_fov_y(mut self, fov: f32) -> Self {
        self.fov_y = fov;

        self
    }
}
