import os
import unittest

from datasets import Dataset

from data_juicer.ops.deduplicator.image_deduplicator import ImageDeduplicator
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ImageDeduplicatorTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    img1_path = os.path.join(data_path, 'img1.png')
    img2_path = os.path.join(data_path, 'img2.jpg')
    img3_path = os.path.join(data_path, 'img3.jpg')
    # img1_dup.png is a duplicate sample of img1.png
    img4_path = os.path.join(data_path, 'img1_dup.png')
    if not os.path.exists(img4_path):
        os.symlink(img1_path, img4_path)
    # img2_dup.jpg is a duplicate sample of img2.jpg
    img5_path = os.path.join(data_path, 'img2_dup.jpg')
    if not os.path.exists(img5_path):
        os.symlink(img2_path, img5_path)
    # img3_dup.jpg is a duplicate sample of img3.jpg
    img6_path = os.path.join(data_path, 'img3_dup.jpg')
    if not os.path.exists(img6_path):
        os.symlink(img3_path, img6_path)
    # img3_dup_dup.jpg is a duplicate sample of img6.jpg
    img7_path = os.path.join(data_path, 'img3_dup_dup.jpg')
    if not os.path.exists(img7_path):
        os.symlink(img6_path, img7_path)

    def _run_image_deduplicator(self, dataset: Dataset, target_list, op):
        key_list = [op.image_key, op.text_key] \
            if op.consider_text else [op.image_key]

        dataset = dataset.map(op.compute_hash)
        dataset, _ = op.process(dataset)
        dataset = dataset.select_columns(column_names=key_list)
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_1(self):

        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        tgt_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator()
        self._run_image_deduplicator(dataset, tgt_list, op)

    def test_2(self):

        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img2_path]
        }]
        tgt_list = [{'images': [self.img1_path]}, {'images': [self.img2_path]}]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator()
        self._run_image_deduplicator(dataset, tgt_list, op)

    def test_3(self):

        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }, {
            'images': [self.img4_path]
        }, {
            'images': [self.img5_path]
        }, {
            'images': [self.img6_path]
        }, {
            'images': [self.img7_path]
        }]
        tgt_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator()
        self._run_image_deduplicator(dataset, tgt_list, op)

    def test_3_consider_text(self):

        ds_list = [{
            'images': [self.img1_path],
            'text': '<video> text1'
        }, {
            'images': [self.img2_path],
            'text': '<video> text2'
        }, {
            'images': [self.img3_path],
            'text': '<video> text3'
        }, {
            'images': [self.img4_path],
            'text': '<video> text1'
        }, {
            'images': [self.img5_path],
            'text': '<video> text5'
        }, {
            'images': [self.img6_path],
            'text': '<video> text3'
        }, {
            'images': [self.img7_path],
            'text': '<video> text7'
        }]
        tgt_list = [{
            'images': [self.img1_path],
            'text': '<video> text1'
        }, {
            'images': [self.img2_path],
            'text': '<video> text2'
        }, {
            'images': [self.img3_path],
            'text': '<video> text3'
        }, {
            'images': [self.img5_path],
            'text': '<video> text5'
        }, {
            'images': [self.img7_path],
            'text': '<video> text7'
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator(consider_text=True)
        self._run_image_deduplicator(dataset, tgt_list, op)

    def test_4(self):

        ds_list = [{
            'images': [self.img1_path, self.img2_path, self.img3_path]
        }, {
            'images': [self.img4_path, self.img5_path, self.img6_path]
        }, {
            'images': [self.img7_path]
        }, {
            'images': [self.img6_path]
        }]
        tgt_list = [{
            'images': [self.img1_path, self.img2_path, self.img3_path]
        }, {
            'images': [self.img7_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator()
        self._run_image_deduplicator(dataset, tgt_list, op)

    def test_4_consider_text(self):

        ds_list = [{
            'images': [self.img1_path, self.img2_path, self.img3_path],
            'text': '<image> text1 <image> text2 <image> text3',
        }, {
            'images': [self.img4_path, self.img5_path, self.img6_path],
            'text': '<image> text1 <image> text5 <image> text3',
        }, {
            'images': [self.img7_path],
            'text': '<image> text6',
        }, {
            'images': [self.img6_path],
            'text': '<image> text6',
        }]
        tgt_list = [{
            'images': [self.img1_path, self.img2_path, self.img3_path],
            'text': '<image> text1 <image> text2 <image> text3',
        }, {
            'images': [self.img4_path, self.img5_path, self.img6_path],
            'text': '<image> text1 <image> text5 <image> text3',
        }, {
            'images': [self.img7_path],
            'text': '<image> text6',
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator(consider_text=True)
        self._run_image_deduplicator(dataset, tgt_list, op)

    def test_5(self):

        ds_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img1_path]
        }, {
            'images': [self.img4_path, self.img5_path]
        }, {
            'images': [self.img7_path, self.img7_path]
        }, {
            'images': [self.img6_path, self.img6_path]
        }]
        tgt_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img1_path]
        }, {
            'images': [self.img7_path, self.img7_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator()
        self._run_image_deduplicator(dataset, tgt_list, op)

    def test_6(self):

        ds_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img1_path]
        }, {
            'images': [self.img4_path, self.img5_path]
        }, {
            'images': [self.img7_path, self.img7_path]
        }, {
            'images': [self.img6_path, self.img6_path]
        }]
        tgt_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img1_path]
        }, {
            'images': [self.img7_path, self.img7_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator(method='dhash')
        self._run_image_deduplicator(dataset, tgt_list, op)

    def test_7(self):

        ds_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img1_path]
        }, {
            'images': [self.img4_path, self.img5_path]
        }, {
            'images': [self.img7_path, self.img7_path]
        }, {
            'images': [self.img6_path, self.img6_path]
        }]
        tgt_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img1_path]
        }, {
            'images': [self.img7_path, self.img7_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator(method='whash')
        self._run_image_deduplicator(dataset, tgt_list, op)

    def test_8(self):

        ds_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img1_path]
        }, {
            'images': [self.img4_path, self.img5_path]
        }, {
            'images': [self.img7_path, self.img7_path]
        }, {
            'images': [self.img6_path, self.img6_path]
        }]
        tgt_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img1_path]
        }, {
            'images': [self.img7_path, self.img7_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator(method='ahash')
        self._run_image_deduplicator(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()
