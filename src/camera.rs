use crate::error::SrResult;
use nalgebra as na;

pub struct Camera {
    position: na::Point3<f32>,
    target: na::Point3<f32>,
    fov: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: na::point![0.0, 0.0, 1.0],
            target: na::point![0.0, 0.0, 0.0],
            fov: 90.0,
        }
    }
}

impl Camera {
    pub fn new(position: na::Point3<f32>, target: na::Point3<f32>, fov: f32) -> SrResult<Self> {
        Ok(Self {
            position,
            target,
            fov,
        })
    }

    pub fn position(&self) -> na::Point3<f32> {
        self.position
    }

    pub fn target(&self) -> na::Point3<f32> {
        self.target
    }

    pub fn fov(&self) -> f32 {
        self.fov
    }
}
