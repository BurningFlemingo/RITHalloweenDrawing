from Scene import *
from RenderTypes import *

WINDOW_WIDTH: int = 1920//2
WINDOW_HEIGHT: int = 1080//2


def main() -> None:
    scene = Scene(Viewport(WINDOW_WIDTH, WINDOW_HEIGHT))
    scene.set_camera(
        Camera(pos=[0, 0, 0.5], target=[0, 0, 1], fov=90, near_plane=0.001, far_plane=100)
    )

    model_transform = Transform(pos=[0, 0, 4], rot=[60, 0, -135])
    
    scene.add_model("assets\\test\\test.obj", model_transform)
    
    light_1_transform = Transform(pos=[0.5, 0.25, 1.5], rot=[60, 0, -135], scale=[0.05, 0.05, 0.05])
    scene.add_light(PointLight(
        light_1_transform.pos, linear_att=0.22, quadratic_att=0.2, color=[1, 0.3, 1.0], specular=[1, 1, 1])
    )
    scene.add_model("assets\\cube\\Cube.obj", light_1_transform)

    light_2_transform = Transform(pos=[-1.5, 0.25, 3], rot=[60, 0, -135], scale=[0.05, 0.05, 0.05])
    scene.add_light(PointLight(
        light_2_transform.pos, linear_att=0.22, quadratic_att=0.2, color=[0.2, 0.2, 1], specular=[1, 1, 1])
    )
    scene.add_model("assets\\cube\\Cube.obj", light_2_transform)

    scene.render()
    scene.present()

    scene.finish()


if __name__ == "__main__":
    main()
