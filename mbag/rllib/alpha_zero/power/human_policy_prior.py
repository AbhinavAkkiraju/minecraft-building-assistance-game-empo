"""
Human policy prior ``pi_h(a | s, g_h)`` for the human-power objective.

Mirrors ``empo.human_policy_prior.HumanPolicyPrior``: a model of what action a
human would take given a hypothetical *possible* goal, independent of any
belief about which goal the human is actually pursuing. This is the Phase-1
(``V_h^m``) input of the human-power theory of Heitzig & Potham
(arXiv:2508.00159), used both to simulate the human during planning and,
in its worst-case form, as the target for the cautious ``Q_h^m``.
"""

from abc import ABC, abstractmethod

import numpy as np

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.goals.possible_goals import MinecraftStructureGoal, PossibleGoal
from mbag.environment.mbag_env import MbagConfigDict, MbagEnv
from mbag.environment.types import CURRENT_BLOCKS, MbagObs


class HumanPolicyPrior(ABC):
    """
    A model of a human's action distribution given a hypothetical possible
    goal ``g_h``.
    """

    def __init__(self, env_config: MbagConfigDict):
        self.env_config = MbagEnv.get_config(env_config)

    @abstractmethod
    def action_distribution(
        self, obs: MbagObs, possible_goal: PossibleGoal
    ) -> np.ndarray:
        """
        Return a probability distribution over actions, shaped
        ``(NUM_CHANNELS, width, height, depth)`` (see
        ``MbagActionDistribution``), for a human pursuing ``possible_goal``
        from ``obs``.
        """


# Per-layer bonus to a helpful action's advantage when the prior respects human
# abilities, so lower layers are built first (with beta_h = 10, each layer down is
# about e^6 times more likely). Without it the one-step heuristic often buries cells
# that still need work and can never finish a goal.
LAYER_PREFERENCE = 0.6


class HeuristicMbagHumanPolicyPrior(HumanPolicyPrior):
    """
    Cheap, untrained human policy prior for a hypothetical possible goal.

    Mirrors the role of ``empo``'s ``HeuristicPotentialPolicy``: a
    potential-based softmax over actions that reduce the mismatch between the
    current build and a possible goal's target structure, computed without
    reference to any assistant model -- worst-case-agnostic by construction,
    since it never conditions on the assistant's behavior at all. This is the
    first (Milestone 1) human policy prior, used to validate the
    ``V_h^e``/``X_h``/``U_r`` formulas end-to-end before investing in a
    learned, genuinely worst-case prior (Milestone 2).

    Simplifications (documented, not hidden): only PLACE_BLOCK/BREAK_BLOCK
    actions are considered -- movement, reach, and inventory constraints are
    ignored, at the same level of simplicity as MBAG's existing
    ``LayerBuilderAgent``/``PriorityQueueAgent`` heuristic baselines. The
    per-cell potential is a strict "does this cell's block id match the
    target" 0/1 indicator, so replacing one wrong block with another wrong
    block scores the same (0) one-step advantage as doing nothing; this
    myopia is left to the learned prior (Milestone 2) to fix if it proves to
    matter in practice.
    """

    def __init__(
        self,
        env_config: MbagConfigDict,
        beta_h: float = 10.0,
        human_cannot_break_bedrock: bool = False,
        lenient: bool = False,
    ):
        """
        Args:
            human_cannot_break_bedrock: Don't consider breaking bedrock (MBAG's
                human players can't), so the modeled human doesn't waste
                actions on it.
            lenient: Match the lenient goal rule: only cells the goal needs
                (non-air target cells) matter, so there is no advantage in
                clearing cells the goal leaves empty.
        """
        super().__init__(env_config)
        self.beta_h = beta_h
        self.human_cannot_break_bedrock = human_cannot_break_bedrock
        self.lenient = lenient

    def action_distribution(
        self, obs: MbagObs, possible_goal: PossibleGoal
    ) -> np.ndarray:
        if not isinstance(possible_goal, MinecraftStructureGoal):
            raise TypeError(
                "HeuristicMbagHumanPolicyPrior only supports MinecraftStructureGoal "
                f"possible goals, got {type(possible_goal)}"
            )

        world_obs, _, _ = obs
        current_block_ids = world_obs[CURRENT_BLOCKS]
        target_block_ids = possible_goal.target.blocks

        width, height, depth = self.env_config["world_size"]
        shape = (MbagActionDistribution.NUM_CHANNELS, width, height, depth)
        advantage = np.zeros(shape)
        valid = np.zeros(shape, dtype=bool)
        valid[0, 0, 0, 0] = True  # NOOP is always an option, with advantage 0.

        is_air = current_block_ids == MinecraftBlocks.AIR
        currently_matches = current_block_ids == target_block_ids

        # PLACE_BLOCK is only physically possible on currently-air cells;
        # placing the target's block id there has advantage 1, any other
        # placeable block id has advantage 0 (a valid but unhelpful move).
        placeable_cells = is_air & ~currently_matches
        for block_id in MinecraftBlocks.PLACEABLE_BLOCK_IDS:
            channel = MbagActionDistribution.PLACE_BLOCK.start + block_id
            valid[channel][placeable_cells] = True
            advantage[channel][placeable_cells & (target_block_ids == block_id)] = 1.0

        # BREAK_BLOCK is only physically possible on non-air cells; it has
        # advantage 1 if the target wants air there (fully resolves the
        # mismatch), 0 if the target wants a different non-air block (a
        # necessary but not immediately rewarded intermediate step).
        breakable_cells = ~is_air & ~currently_matches
        if self.human_cannot_break_bedrock:
            breakable_cells &= current_block_ids != MinecraftBlocks.BEDROCK
        valid[MbagActionDistribution.BREAK_BLOCK][breakable_cells] = True
        if not self.lenient:
            advantage[MbagActionDistribution.BREAK_BLOCK][
                breakable_cells & (target_block_ids == MinecraftBlocks.AIR)
            ] = 1.0

        if self.lenient:
            # Breaking a wrong block in a cell the goal needs is a necessary step
            # toward the goal, so count it as progress too.
            advantage[MbagActionDistribution.BREAK_BLOCK][
                breakable_cells & (target_block_ids != MinecraftBlocks.AIR)
            ] = 1.0

        if self.human_cannot_break_bedrock or self.lenient:
            # Build bottom-up, as a sensible human would: prefer lower layers, so a
            # cell that still needs replacing isn't buried under new blocks (after
            # which it can't be reached any more).
            # A bonus for lower layers (rather than a penalty for higher ones) keeps
            # every helpful action better than doing nothing.
            layers_from_top = (height - 1 - np.arange(height, dtype=float))[
                None, None, :, None
            ]
            advantage = np.where(
                advantage > 0, advantage + LAYER_PREFERENCE * layers_from_top, advantage
            )

            # Respect the human's abilities: only consider actions the environment
            # actually allows (e.g. placing next to an existing block), so the
            # modeled human doesn't waste steps on moves that do nothing.
            world_obs, inventory_obs, timestep = obs
            action_mask = MbagActionDistribution.get_mask(
                self.env_config,
                (world_obs[None], inventory_obs[None], np.asarray(timestep)[None]),
            )[0]
            valid &= action_mask
            valid[0, 0, 0, 0] = True

        logits = np.where(valid, self.beta_h * advantage, -np.inf)
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= probs.sum()
        return probs
