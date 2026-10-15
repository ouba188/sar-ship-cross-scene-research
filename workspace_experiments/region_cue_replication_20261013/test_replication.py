import unittest
import numpy as np
import pandas as pd
from summarize_replication import equal_task_summary
from evaluate_replication import scope_masks


class ReplicationTests(unittest.TestCase):
    def test_equal_ports_not_row_weighted(self):
        f = pd.DataFrame(dict(dataset=["ports"]*7, task=["large"]*5+["small"]*2,
            scope=["full"]*7, value=[0.]*5+[100.]*2))
        task, result = equal_task_summary(f, ["scope"], ["value"])
        self.assertEqual(len(task), 2)
        self.assertEqual(result.value.iloc[0], 50.)

    def test_invalid_and_support_rows_and_unknown_identities(self):
        frame = pd.DataFrame(dict(identity=["123456789", "123456789", "", "", "987654321"],
                                  dataset_name=["ports"]*5))
        control = pd.DataFrame(dict(region_valid=[True, True, True, False, True], sham_valid=[True]*5))
        data = dict(support=np.array([[0]]), eval_masks=np.array([[False, True, True, True, True]]))
        masks = scope_masks(frame, control, data, 0)
        np.testing.assert_array_equal(masks["full"], [False, True, True, False, True])
        np.testing.assert_array_equal(masks["unseen_support_vessel"], [False, False, False, False, True])


if __name__ == "__main__":
    unittest.main()
