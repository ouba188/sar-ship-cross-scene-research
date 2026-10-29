"""Use current pre-M3 task features for the deep comparison stream only."""
from sar_m123.selective_matching import SelectiveReferenceMatch


class TaskAnchorMatch(SelectiveReferenceMatch):
    def forward(self, x, anchor, local, slots, local_slots, valid, energy,
                local_energy, reference, pair_valid, return_details=False):
        current_reference = dict(reference, anchor=reference['features'].detach())
        return super().forward(x, x.detach(), local, slots, local_slots, valid,
                               energy, local_energy, current_reference, pair_valid,
                               return_details=return_details)


def attach(net):
    for blocks in net.m3:
        for p, old in enumerate(blocks):
            new = TaskAnchorMatch(channels=old.project.in_channels,
                                 dynamic=old.dynamic, frequency=old.frequency,
                                 temperature=old.temperature).to(old.gain.device)
            new.load_state_dict(old.state_dict(), strict=True)
            blocks[p] = new.eval().requires_grad_(False)
    return net
