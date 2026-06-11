//! Bevy + sunray example.
//!
//! Builds a Bevy `App` from the **minimal** plugin set (Option B in
//! `docs/bevy_integration.md`) — no `RenderPlugin`, no wgpu — and adds
//! [`SunrayRenderPlugin`] + [`SunrayEguiPlugin`] to render the sunray
//! ray-tracer into the Bevy-managed window.
//!
//! Controls: hold **right mouse** to look, **WASD** to move, **Space/Ctrl** up/down,
//! **Shift** to go faster.
//!
//! Run with:
//! ```text
//! cargo run --release --features bevy --example bevy_app
//! ```
//!
//! egui is fully wired: input + tessellation + extract + a heap+Slang GPU paint
//! pass that overlays the egui window on top of the ray-traced frame (see
//! docs/bevy_integration.md §6).

use bevy_a11y::AccessibilityPlugin;
use bevy_app::{App, Startup, TaskPoolPlugin, Update};
use bevy_asset::{AssetPlugin, Assets};
use bevy_diagnostic::FrameCountPlugin;
use bevy_ecs::prelude::*;
use bevy_input::ButtonInput;
use bevy_input::InputPlugin;
use bevy_input::keyboard::KeyCode;
use bevy_input::mouse::{MouseButton, MouseMotion};
use bevy_log::LogPlugin;
use bevy_math::primitives::Cuboid;
use bevy_math::{EulerRot, Quat, Vec2, Vec3};
use bevy_render::mesh::Mesh;
use bevy_time::{Time, TimePlugin};
use bevy_transform::TransformPlugin;
use bevy_transform::components::Transform;
use bevy_window::{Window, WindowPlugin};
use bevy_winit::WinitPlugin;

use sunray::bevy_integration::{
    EguiContext, SunrayCamera, SunrayEguiPlugin, SunrayMaterial, SunrayMeshInstance, SunrayRenderPlugin, SunrayScene,
};

/// Simple fly-camera state stored next to the camera's `Transform`.
#[derive(Component)]
struct FlyCam {
    yaw: f32,
    pitch: f32,
}

fn main() {
    App::new()
        // Minimal Bevy: ECS + windowing + input + time + transforms. No render stack.
        .add_plugins((
            LogPlugin::default(),
            TaskPoolPlugin::default(),
            // Assets so `SunrayMeshInstance` can reference `Mesh` assets;
            // `SunrayRenderPlugin` registers `Assets<Mesh>` itself.
            AssetPlugin::default(),
            FrameCountPlugin,
            TimePlugin,
            InputPlugin,
            WindowPlugin {
                primary_window: Some(Window {
                    title: "sunray + bevy".into(),
                    ..Default::default()
                }),
                ..Default::default()
            },
            AccessibilityPlugin,
            WinitPlugin::default(),
            TransformPlugin,
        ))
        // Our custom ash renderer + egui input/extract layer.
        .add_plugins((SunrayRenderPlugin::default(), SunrayEguiPlugin))
        // Ask the renderer to load this glTF once the device exists.
        .insert_resource(SunrayScene::with_gltf("examples/assets/Room.glb"))
        .add_systems(Startup, setup)
        .add_systems(Update, (fly_cam, ui_system))
        .run();
}

fn setup(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    // The camera is an ordinary ECS entity: a Transform + SunrayCamera marker.
    commands.spawn((
        Transform::from_xyz(0.0, 2.0, 10.0),
        SunrayCamera { fov_y_degrees: 45.0 },
        FlyCam { yaw: 0.0, pitch: 0.0 },
    ));

    // Runtime mesh assets → BLASes built on the fly, rendered on top of the
    // glTF scene. The material is per mesh *asset* (one BLAS each), so the two
    // entities sharing `cube` share the red material.
    let cube = meshes.add(Mesh::from(Cuboid::new(1.0, 1.0, 1.0)));
    let red = SunrayMaterial {
        base_color: [0.8, 0.15, 0.15, 1.0],
        roughness: 0.4,
        ..Default::default()
    };
    commands.spawn((Transform::from_xyz(2.0, 0.5, 0.0), SunrayMeshInstance { mesh: cube.clone() }, red));
    commands.spawn((Transform::from_xyz(-2.0, 0.5, 0.0), SunrayMeshInstance { mesh: cube }, red));

    // A small emissive cube acting as an extra light (NEE picks up its
    // triangles through the runtime emissive-triangle path).
    let lamp = meshes.add(Mesh::from(Cuboid::new(0.3, 0.3, 0.3)));
    commands.spawn((
        Transform::from_xyz(0.0, 2.5, 0.0),
        SunrayMeshInstance { mesh: lamp },
        SunrayMaterial {
            emissive: [1.0, 0.9, 0.7],
            emissive_strength: 20.0,
            ..Default::default()
        },
    ));
}

fn fly_cam(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion: MessageReader<MouseMotion>,
    mut query: Query<(&mut Transform, &mut FlyCam)>,
) {
    let Ok((mut transform, mut cam)) = query.single_mut() else {
        return;
    };
    let dt = time.delta_secs();

    // Mouse look while the right button is held.
    if buttons.pressed(MouseButton::Right) {
        let mut delta = Vec2::ZERO;
        for ev in motion.read() {
            delta += ev.delta;
        }
        cam.yaw -= delta.x * 0.002;
        cam.pitch -= delta.y * 0.002;
        let limit = std::f32::consts::FRAC_PI_2 - 0.01;
        cam.pitch = cam.pitch.clamp(-limit, limit);
    } else {
        motion.clear();
    }
    transform.rotation = Quat::from_euler(EulerRot::YXZ, cam.yaw, cam.pitch, 0.0);

    let forward = *transform.forward();
    let right = *transform.right();
    let mut dir = Vec3::ZERO;
    if keys.pressed(KeyCode::KeyW) {
        dir += forward;
    }
    if keys.pressed(KeyCode::KeyS) {
        dir -= forward;
    }
    if keys.pressed(KeyCode::KeyD) {
        dir += right;
    }
    if keys.pressed(KeyCode::KeyA) {
        dir -= right;
    }
    if keys.pressed(KeyCode::Space) {
        dir += Vec3::Y;
    }
    if keys.pressed(KeyCode::ControlLeft) {
        dir -= Vec3::Y;
    }

    let speed = if keys.pressed(KeyCode::ShiftLeft) { 9.0 } else { 3.0 };
    if dir != Vec3::ZERO {
        transform.translation += dir.normalize() * speed * dt;
    }
}

fn ui_system(egui_ctx: Res<EguiContext>) {
    egui::Window::new("sunray + bevy").show(egui_ctx.ctx(), |ui| {
        ui.label("Hardware ray tracing, driven by Bevy ECS.");
        ui.separator();
        ui.label("Right mouse: look · WASD: move · Space/Ctrl: up/down · Shift: faster");
        ui.separator();
        ui.label("This panel is painted by a heap+Slang egui pass over the RT frame.");
    });
}
