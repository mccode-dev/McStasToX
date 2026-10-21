# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Mccode-dev contributors (https://github.com/mccode-dev)

from mcstastox.LoadFile import Transfer


def test_transfer_uses_mccode_and_scipp_names() -> None:
    transfer = Transfer("t", "time")

    assert transfer.mcstas_variable == "t"
    assert transfer.scipp_coord == "time"
    assert transfer.unit is None


def test_transfer_accepts_unit() -> None:
    transfer = Transfer("t", "time", "s")

    assert transfer.unit == "s"
