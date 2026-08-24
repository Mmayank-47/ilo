import unittest
from data.dataset import parse_ravdess_filename, split_by_actor, get_speaker_aware_kfolds

class TestDataset(unittest.TestCase):
    def test_parse_ravdess_filename(self):
        filename = "03-01-05-01-02-01-12.wav"
        meta = parse_ravdess_filename(filename)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["modality"], 3)
        self.assertEqual(meta["vocal_channel"], 1)
        self.assertEqual(meta["emotion_code"], "05")
        self.assertEqual(meta["intensity"], 1)
        self.assertEqual(meta["statement"], 2)
        self.assertEqual(meta["repetition"], 1)
        self.assertEqual(meta["actor_id"], 12)

        # Invalid filename format
        invalid_filename = "invalid_file.wav"
        meta_invalid = parse_ravdess_filename(invalid_filename)
        self.assertIsNone(meta_invalid)

    def test_split_by_actor(self):
        samples = [
            {"filepath": "path1", "actor_id": 1, "label_idx": 0},
            {"filepath": "path2", "actor_id": 2, "label_idx": 1},
            {"filepath": "path3", "actor_id": 23, "label_idx": 0},
            {"filepath": "path4", "actor_id": 24, "label_idx": 3},
        ]
        trainval, test = split_by_actor(samples, [23, 24])
        self.assertEqual(len(trainval), 2)
        self.assertEqual(len(test), 2)
        self.assertEqual([s["actor_id"] for s in trainval], [1, 2])
        self.assertEqual([s["actor_id"] for s in test], [23, 24])

    def test_get_speaker_aware_kfolds(self):
        # Create synthetic list of samples representing multiple actors and classes
        samples = []
        # 10 actors, multiple samples per actor
        for actor_id in range(1, 11):
            for emotion_idx in range(4):
                samples.append({
                    "filepath": f"actor_{actor_id}_emo_{emotion_idx}.wav",
                    "actor_id": actor_id,
                    "label_idx": emotion_idx
                })
        
        folds = get_speaker_aware_kfolds(samples, n_folds=3, seed=42)
        self.assertEqual(len(folds), 3)

        for train_idx, val_idx in folds:
            train_actors = {samples[i]["actor_id"] for i in train_idx}
            val_actors = {samples[i]["actor_id"] for i in val_idx}
            
            # Intersection of train and val actors must be empty (speaker-aware split)
            self.assertEqual(len(train_actors.intersection(val_actors)), 0)
            self.assertTrue(len(train_idx) > 0)
            self.assertTrue(len(val_idx) > 0)

if __name__ == "__main__":
    unittest.main()
