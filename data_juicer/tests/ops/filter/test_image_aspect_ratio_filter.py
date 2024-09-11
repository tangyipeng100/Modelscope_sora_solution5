import os
import unittest

from datasets import Dataset

from data_juicer.ops.filter.image_aspect_ratio_filter import \
    ImageAspectRatioFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ImageAspectRatioFilterTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    img1_path = os.path.join(data_path, 'img1.png')
    img2_path = os.path.join(data_path, 'img2.jpg')
    img3_path = os.path.join(data_path, 'img3.jpg')

    def _run_image_aspect_ratio_filter(self, dataset: Dataset, target_list,
                                       op):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats)
        dataset = dataset.filter(op.process)
        dataset = dataset.select_columns(column_names=[op.image_key])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_filter1(self):

        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        tgt_list = [{'images': [self.img1_path]}]
        dataset = Dataset.from_list(ds_list)
        op = ImageAspectRatioFilter(min_ratio=0.8, max_ratio=1.2)
        self._run_image_aspect_ratio_filter(dataset, tgt_list, op)

    def test_filter2(self):

        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        tgt_list = [{'images': [self.img1_path]}, {'images': [self.img2_path]}]
        dataset = Dataset.from_list(ds_list)
        op = ImageAspectRatioFilter(min_ratio=0.8)
        self._run_image_aspect_ratio_filter(dataset, tgt_list, op)

    def test_filter3(self):

        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        tgt_list = [{'images': [self.img1_path]}, {'images': [self.img3_path]}]
        dataset = Dataset.from_list(ds_list)
        op = ImageAspectRatioFilter(max_ratio=1.2)
        self._run_image_aspect_ratio_filter(dataset, tgt_list, op)

    def test_any(self):

        ds_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img3_path]
        }, {
            'images': [self.img1_path, self.img3_path]
        }]
        tgt_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img1_path, self.img3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageAspectRatioFilter(min_ratio=0.8,
                                    max_ratio=1.2,
                                    any_or_all='any')
        self._run_image_aspect_ratio_filter(dataset, tgt_list, op)

    def test_all(self):

        ds_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img3_path]
        }, {
            'images': [self.img1_path, self.img3_path]
        }]
        tgt_list = []
        dataset = Dataset.from_list(ds_list)
        op = ImageAspectRatioFilter(min_ratio=0.8,
                                    max_ratio=1.2,
                                    any_or_all='all')
        self._run_image_aspect_ratio_filter(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()
