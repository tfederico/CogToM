from minigrid.core.world_object import WorldObj, fill_coords, point_in_rect
from minigrid.core.constants import COLORS


class Goal(WorldObj):
    def __init__(self, color="green"):
        super().__init__("goal", color)

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])
