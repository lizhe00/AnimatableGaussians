from screeninfo import get_monitors


def get_monitor_count():
    try:
        count = len(get_monitors())
    except:
        count = 0
    return count


if get_monitor_count() > 0:
    print('# Using GL-based Renderer')
    from .renderer_gl import Renderer
else:
    print('# Using Pytorch3d Renderer')
    from .renderer_pytorch3d import Renderer
