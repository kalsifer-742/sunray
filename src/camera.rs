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
            fov: 45.0,
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

    pub fn set_position(mut self, position: na::Point3<f32>) -> Self {
        self.position = position;

        self
    }

    pub fn set_target(mut self, target: na::Point3<f32>) -> Self {
        self.target = target;

        self
    }

    pub fn set_fov(mut self, fov: f32) -> Self {
        self.fov = fov;

        self
    }
}
