from Scene import *
from RenderTypes import *

WINDOW_WIDTH: int = 1920 // 2
WINDOW_HEIGHT: int = 1080 // 2


def main() -> None:
    scene = Scene(Viewport(WINDOW_WIDTH, WINDOW_HEIGHT))
    camera = Camera(pos=[0, -0.0, 1.0], target=[0, 0, 5],
                    fov=70, near_plane=0.01, far_plane=12)
    scene.set_camera(
        camera
    )

    model_transform = Transform(pos=[0, 0, 8], rot=[80, 20, -110])

    scene.add_model("assets\\normal_map_example\\normal_map_example.obj", model_transform)

    # scene.add_light(PointLight(pos=[0, 5, 0], color=[1.0, 1.0, 1.0], intensity=0.8))

    cube_transform = Transform(
        pos=Vec3(0.5, 0.5, 2.0), rot=[0, 0, 0], scale=[0.025, 0.025, 0.025])
    scene.add_light(SpotLight(
        pos=cube_transform.pos, dir=[-1.0, 0.0, 1.0],
        inner_cutoff_angle=10.0, outer_cutoff_angle=60.0,
        color=[1.0, 0.3, 1], intensity=50.0)
    )
    scene.add_model("assets\\cube\\Cube.obj", cube_transform)

    scene.render()
    scene.present()

    scene.finish()


if __name__ == "__main__":
    main()
