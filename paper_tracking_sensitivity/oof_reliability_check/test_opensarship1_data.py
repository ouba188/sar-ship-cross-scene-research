"""Native XML joins must not depend on sorted chip order or coarse filenames."""

from pathlib import Path
import tempfile
import unittest

from opensarship1_data import read_native
from opensarship2_labels import map_opensarship2_labels


class NativeJoinTest(unittest.TestCase):
    def test_center_join_and_fine_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp)/'S1A_IW_GRDH_TEST'
            chips = scene/'Patch_Uint8'
            chips.mkdir(parents=True)
            for center in ('x100_y200', 'x20_y30'):
                for pol in ('vv', 'vh'):
                    (chips/f'Visual_Cargo_{center}_{pol}.tif').touch()
            xml = '<ShipList>'
            for x, y, mmsi, label in [(20, 30, 111, 'Bulk Carrier'), (100, 200, 222, 'Cargo')]:
                xml += (f'<ship><SARShipInformation><Center_x>{x}</Center_x><Center_y>{y}</Center_y>'
                    f'</SARShipInformation><AISShipInformation><MMSI>{mmsi}</MMSI></AISShipInformation>'
                    f'<MarineTrafficInformation><Elaborated_type>{label}</Elaborated_type>'
                    '</MarineTrafficInformation></ship>')
            (scene/'Ship.xml').write_text(xml+'</ShipList>')
            result = map_opensarship2_labels(read_native(Path(tmp)))
            self.assertIn('x20_y30', result.iloc[0].chip_path_vv)
            self.assertEqual(result.iloc[0].mmsi_or_ship_id, '111')
            self.assertEqual(result.iloc[0].class_name, 'bulk_carrier')
            self.assertFalse(result.iloc[1].label_included)
            self.assertTrue(result.image_available.all())


if __name__ == '__main__':
    unittest.main()
