import unittest

import torch
import torch.nn.functional as F

from model.background_ambiguity_soft import continuous_background_loss


class ContinuousPBATests(unittest.TestCase):
    def example(self):
        logits = torch.randn(1, 4, 1, 4, requires_grad=True)
        labels = torch.tensor([[[0, 1, 3, 255]]])
        cams = torch.full((1, 3, 1, 4), .8, requires_grad=True)
        auxiliary = torch.full((1, 3, 1, 4), .6, requires_grad=True)
        return dict(logits=logits, labels=labels, cams=cams, cams_aux=auxiliary,
                    image_tags=torch.ones(1, 3), valid_mask=labels != 255, old_classes=2)

    def test_no_cam_evidence_recovers_original_ce_and_gradient(self):
        values = self.example()
        values['cams'] = values['cams'] * 0
        loss, stats = continuous_background_loss(**values)
        original = F.cross_entropy(values['logits'], values['labels'], ignore_index=255)
        torch.testing.assert_close(loss, original)
        torch.testing.assert_close(torch.autograd.grad(loss, values['logits'], retain_graph=True)[0],
                                   torch.autograd.grad(original, values['logits'])[0])
        self.assertEqual(stats['ambiguity_mass'], 0)

    def test_absent_new_image_tag_recovers_original_ce(self):
        values = self.example()
        values['image_tags'][0, 2] = 0
        actual, _ = continuous_background_loss(**values)
        torch.testing.assert_close(actual, F.cross_entropy(values['logits'], values['labels'], ignore_index=255))

    def test_all_foreground_and_ignored_pixel_gradients_are_unchanged(self):
        values = self.example()
        loss, stats = continuous_background_loss(**values)
        original = F.cross_entropy(values['logits'], values['labels'], ignore_index=255)
        actual_gradient, = torch.autograd.grad(loss, values['logits'], retain_graph=True)
        original_gradient, = torch.autograd.grad(original, values['logits'])
        torch.testing.assert_close(actual_gradient[..., 1:], original_gradient[..., 1:])
        self.assertFalse(stats['thresholds_used'])
        self.assertFalse(stats['prototypes_used'])

    def test_cam_and_student_confidence_weights_are_detached(self):
        values = self.example()
        loss, _ = continuous_background_loss(**values)
        loss.backward()
        self.assertTrue(torch.isfinite(values['logits'].grad).all())
        self.assertIsNone(values['cams'].grad)
        self.assertIsNone(values['cams_aux'].grad)

    def test_cam_evidence_is_continuous(self):
        values = self.example()
        losses = []
        for evidence in (.49, .5, .51):
            cams = torch.full_like(values['cams'], evidence)
            loss, _ = continuous_background_loss(**dict(values, cams=cams, cams_aux=cams))
            losses.append(loss)
        torch.testing.assert_close(losses[1], (losses[0] + losses[2]) / 2)

    def test_all_padding_has_zero_loss_and_gradient(self):
        values = self.example()
        values['valid_mask'].zero_()
        loss, _ = continuous_background_loss(**values)
        self.assertEqual(float(loss), 0)
        loss.backward()
        self.assertEqual(float(values['logits'].grad.abs().sum()), 0)

    def test_weighted_ce_uses_original_valid_pixel_denominator(self):
        values = self.example()
        values['cams'] = values['cams'] * 0
        weight = torch.tensor([.1, 1., 1., 1.])
        actual, _ = continuous_background_loss(**values, class_weight=weight)
        expected = F.cross_entropy(values['logits'], values['labels'], weight=weight, ignore_index=255,
                                   reduction='sum') / 3
        torch.testing.assert_close(actual, expected)

    def test_multiple_new_classes_use_continuous_evidence_distribution(self):
        logits = torch.tensor([[[[0.]], [[1.]], [[.3]], [[.7]]]], requires_grad=True)
        cams = torch.tensor([[[[.5]], [[.8]], [[.4]]]])
        loss, _ = continuous_background_loss(logits, torch.zeros(1, 1, 1, dtype=torch.long),
            cams, cams, torch.ones(1, 3), torch.ones(1, 1, 1, dtype=torch.bool), old_classes=1)
        logp = logits.log_softmax(1)
        confidence = logp[:, 2:].detach().exp()
        partial = -torch.logaddexp(logp[:, :1], logp[:, 2:])
        expected_candidates = ((1 - confidence) * partial - confidence * logp[:, 2:])
        expected = .2 * -logp[:, 0] + .8 * (expected_candidates[:, 0] * (2 / 3) + expected_candidates[:, 1] / 3)
        torch.testing.assert_close(loss, expected.squeeze())

    def test_zero_strength_recovers_ce_with_positive_cam_evidence(self):
        values = self.example()
        loss, _ = continuous_background_loss(**values, strength=0)
        baseline = F.cross_entropy(values['logits'], values['labels'], ignore_index=255)
        torch.testing.assert_close(loss, baseline)
        torch.testing.assert_close(torch.autograd.grad(loss, values['logits'], retain_graph=True)[0],
                                   torch.autograd.grad(baseline, values['logits'])[0])

    def test_half_strength_interpolates_loss_and_gradient_preserving_foreground(self):
        values = self.example()
        full, full_stats = continuous_background_loss(**values)
        half, half_stats = continuous_background_loss(**values, strength=.5)
        baseline = F.cross_entropy(values['logits'], values['labels'], ignore_index=255)
        torch.testing.assert_close(half, (full + baseline) / 2)
        full_gradient, = torch.autograd.grad(full, values['logits'], retain_graph=True)
        half_gradient, = torch.autograd.grad(half, values['logits'], retain_graph=True)
        baseline_gradient, = torch.autograd.grad(baseline, values['logits'])
        torch.testing.assert_close(half_gradient, (full_gradient + baseline_gradient) / 2)
        torch.testing.assert_close(half_gradient[..., 1:], baseline_gradient[..., 1:])
        self.assertAlmostEqual(half_stats['mean_directional_weight'], full_stats['mean_directional_weight'] / 2)

    def test_invalid_strength_is_rejected(self):
        for strength in (-.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                continuous_background_loss(**self.example(), strength=strength)


if __name__ == '__main__':
    unittest.main()
