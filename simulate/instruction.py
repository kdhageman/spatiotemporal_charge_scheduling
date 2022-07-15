from enum import Enum


class InstructionType(Enum):
    move = 1
    wait = 2
    charge = 3


class MoveInstruction:
    def __init__(self, node):
        self.node = node
        self.type = InstructionType.move


class WaitInstruction:
    def __init__(self, node, t):
        self.node = node
        self.t = t
        self.type = InstructionType.wait


class ChargeInstruction:
    def __init__(self, node, t):
        self.node = node
        self.t = t
        self.type = InstructionType.charge
