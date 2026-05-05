from __future__ import annotations

import unittest


class UISmokeTests(unittest.TestCase):
    def test_app_module_imports(self) -> None:
        try:
            import dearpygui.dearpygui as dpg  # noqa: F401
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise unittest.SkipTest("dearpygui is not installed") from exc
        from ct_to_bmd_studio.ui.app import StudioApp

        app = StudioApp()
        self.assertEqual(app.state.mode_config.mode, "single")

    def test_ui_setters_coerce_none_to_empty_strings(self) -> None:
        try:
            import dearpygui.dearpygui as dpg  # noqa: F401
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise unittest.SkipTest("dearpygui is not installed") from exc
        from ct_to_bmd_studio.ui.app import StudioApp

        app = StudioApp()
        app.set_ct_filename(None)  # type: ignore[arg-type]
        app.set_existing_seg(None)  # type: ignore[arg-type]
        app.set_dataset_root(None)  # type: ignore[arg-type]
        app.set_batch_root(None)  # type: ignore[arg-type]
        app.set_label_input(None)  # type: ignore[arg-type]
        app.set_segmentation_source(None)  # type: ignore[arg-type]
        self.assertEqual(app.state.mode_config.ct_filename, "")
        self.assertEqual(app.state.mode_config.existing_seg_filename, "")
        self.assertEqual(app.state.mode_config.dataset_root, "")
        self.assertEqual(app.state.mode_config.batch_root, "")
        self.assertEqual(app.state.label_input_text, "")
        self.assertEqual(app.state.mode_config.segmentation_source, "totalsegmentator")

    def test_totalseg_toggle_keeps_multiple_labels(self) -> None:
        try:
            import dearpygui.dearpygui as dpg  # noqa: F401
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise unittest.SkipTest("dearpygui is not installed") from exc
        from ct_to_bmd_studio.ui.app import StudioApp

        app = StudioApp()
        app.state.mode_config.selected_totalseg_labels = []
        app.toggle_totalseg_label("femur_left", True)
        app.toggle_totalseg_label("femur_right", True)
        self.assertEqual(app.state.mode_config.selected_totalseg_labels, ["femur_left", "femur_right"])

    def test_totalseg_checkboxes_store_label_in_user_data(self) -> None:
        try:
            import dearpygui.dearpygui as dpg
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise unittest.SkipTest("dearpygui is not installed") from exc
        from ct_to_bmd_studio.ui.app import StudioApp
        from ct_to_bmd_studio.ui.windows import segmentation_setup

        dpg.create_context()
        try:
            app = StudioApp()
            with dpg.window(tag="test_root"):
                pass
            segmentation_setup.render(app, "test_root")
            self.assertEqual(dpg.get_item_user_data(app._totalseg_label_tag("femur_left")), "femur_left")
        finally:
            dpg.destroy_context()


if __name__ == "__main__":
    unittest.main()
