"""
Progress toward a set of reference goals, used as a potential for shaping X_h.

``progress(s, g)`` is 0 in the initial world, 1 exactly when the world matches the
goal ``g`` (block for block), and in between rises with every correct block and
falls with every wrong block:

    progress = (correct - correct0 - (wrong - wrong0)) / (goal_blocks - correct0 + wrong0)

where ``correct`` counts non-air goal cells that match, ``wrong`` counts non-air cells
that differ from the goal, and the ``*0`` values are the same counts in the initial
world. The result is clipped to [0, 1].

The shaped aggregate goal ability starts from ``mean_g progress(s, g) ** zeta`` and
the network learns a correction on top of it. The regression targets are unchanged
(0/1 achievement), so the learned X_h still converges to the same quantity; the
potential only gives it a dense, exact signal about progress, so that destroying
(or building) blocks changes X_h immediately.
"""

import torch

from mbag.environment.blocks import MinecraftBlocks


class ProgressReference:
    def __init__(self, goals: torch.Tensor, initial_world: torch.Tensor):
        """
        Args:
            goals: Long tensor (N, W, H, D) of reference goals, in the same encoding
                (full world, including floor) as the environment's goal blocks.
            initial_world: Long tensor (W, H, D) of the world at the start of an
                episode (floor only).
        """
        goals = goals.long()
        initial_world = initial_world.long()
        goal_nonair = goals != MinecraftBlocks.AIR
        initial_nonair = initial_world != MinecraftBlocks.AIR
        self.goals = goals
        self.goal_nonair = goal_nonair
        self.correct0 = (
            ((initial_world[None] == goals) & goal_nonair).flatten(1).sum(1).float()
        )
        self.wrong0 = (
            (initial_nonair[None] & (initial_world[None] != goals))
            .flatten(1)
            .sum(1)
            .float()
        )
        self.denominator = (
            goal_nonair.flatten(1).sum(1).float() - self.correct0 + self.wrong0
        ).clamp(min=1.0)

    def progress(self, current_blocks: torch.Tensor) -> torch.Tensor:
        """
        Progress of each world in the batch toward each reference goal.

        Args:
            current_blocks: Long tensor (B, W, H, D).

        Returns:
            Float tensor (B, N) in [0, 1].
        """
        device = current_blocks.device
        goals = self.goals.to(device)
        goal_nonair = self.goal_nonair.to(device)
        current = current_blocks.long()[:, None]  # (B, 1, W, H, D)
        matches = current == goals[None]  # (B, N, W, H, D)
        correct = (matches & goal_nonair[None]).flatten(2).sum(2).float()
        wrong = (
            ((current != MinecraftBlocks.AIR) & ~matches).flatten(2).sum(2).float()
        )
        numerator = (correct - self.correct0.to(device)) - (
            wrong - self.wrong0.to(device)
        )
        return (numerator / self.denominator.to(device)).clamp(0.0, 1.0)


class StepsToGoalReference:
    """
    A potential for each reference goal that approximates V_h^e(s, g): the chance
    the human reaches g if it pursued g from s, for use with the lenient goal rule
    (only the cells a goal needs matter).

        potential(s, g) = 0                      if bedrock sits on a cell g needs
                        = gamma_h ** steps(s, g)  otherwise,

    where steps(s, g) is the number of steps the human still needs: one per needed
    cell that is empty and two per needed cell holding a wrong (breakable) block.
    It is 1 exactly when every needed cell is correct. Clearing the last bedrock
    block on a goal's cells raises its potential from 0, which gives a graded signal
    for removing obstacles even before the human has made any progress.
    """

    def __init__(self, goals: torch.Tensor, gamma: float):
        """
        Args:
            goals: Long tensor (N, W, H, D) of reference goals, in the same encoding
                (full world, including floor) as the environment's goal blocks.
            gamma: The human's discount factor gamma_h.
        """
        self.goals = goals.long()
        self.needed = self.goals != MinecraftBlocks.AIR
        self.gamma = gamma

    def progress(self, current_blocks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            current_blocks: Long tensor (B, W, H, D).

        Returns:
            Float tensor (B, N) in [0, 1].
        """
        device = current_blocks.device
        goals = self.goals.to(device)[None]  # (1, N, W, H, D)
        needed = self.needed.to(device)[None]
        current = current_blocks.long()[:, None]  # (B, 1, W, H, D)
        correct = current == goals
        blocked = (
            ((current == MinecraftBlocks.BEDROCK) & needed & ~correct)
            .flatten(2)
            .any(2)
        )
        empty = (current == MinecraftBlocks.AIR) & needed & ~correct
        wrong = (current != MinecraftBlocks.AIR) & needed & ~correct
        steps = empty.flatten(2).sum(2).float() + 2 * wrong.flatten(2).sum(2).float()
        potential = torch.pow(torch.tensor(self.gamma, device=device), steps)
        return torch.where(blocked, torch.zeros_like(potential), potential)
