# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Mccode-dev contributors (https://github.com/mccode-dev)

from mcstastox.ReadNeXus import _MCSTAS_VERSION_SETTINGS, _get_mcstas_version_settings


def test_mcstas_version_settings_loading() -> None:
    setting_2_7_0 = _MCSTAS_VERSION_SETTINGS[(2, 7, 0)]
    setting_3_5_20 = _MCSTAS_VERSION_SETTINGS[(3, 5, 20)]
    assert _get_mcstas_version_settings((2, 8, 0)) == setting_2_7_0
    assert _get_mcstas_version_settings((3, 2, 0)) == setting_2_7_0
    assert _get_mcstas_version_settings((3, 5, 20)) == setting_3_5_20
    assert _get_mcstas_version_settings((3, 6, 0)) == setting_3_5_20
