from Scene import *
from RenderTypes import *

WINDOW_WIDTH: int = 1920//2
WINDOW_HEIGHT: int = 1080//2


def main() -> None:
    scene = Scene(Viewport(WINDOW_WIDTH, WINDOW_HEIGHT))
    camera = Camera(pos=[0, 0, 0.5], target=[0, 0, 1], fov=90, near_plane=0.001, far_plane=100)
    scene.set_camera(
        camera
    )

    model_transform = Transform(pos=[0, 0, 4], rot=[60, 0, -135])
    
    scene.add_model("assets\\test\\test.obj", model_transform)

    scene.add_light(PointLight(
        pos=[2, 0, 3], color=[1.0, 0.0, 1], intensity=1.0)
    )

    scene.add_light(DirectionalLight(dir=[0, -1, 0.5], color=[1.0, 1.0, 1.0], intensity=0.5))

    scene.render()
    scene.present()

    scene.finish()


if __name__ == "__main__":
    main()
