import numpy as np

class Sampler:
    """
    Implement method `process`
    """

    @staticmethod
    def process(x, distr):
        """
        Choose elements from `x` to satisfy `distr` distribution.

        :param x: A list of numbers to choose from
        :param distr: The expected normalized histogram of the output
        :return: A selection of elements from `x`
        """

        # Konversi ke numpy array untuk efisiensi
        x = np.array(x)

        # Step 1: Hapus Outliers
        mean_x = np.mean(x)
        std_x = np.std(x)
        x = x[np.abs(x - mean_x) <= 2 * std_x]

        # Step 2: Tentukan bins sesuai distribusi yang diinginkan
        num_bins = len(distr)
        bins = np.linspace(np.min(x), np.max(x), num_bins + 1)
        bin_indices = np.digitize(x, bins) - 1

        # Step 3: Hitung jumlah elemen yang harus diambil dari setiap bin
        total_count = len(x)
        expected_counts = np.round(distr * total_count).astype(int)

        # Step 4: Sampling elemen dari x berdasarkan distribusi `distr`
        y = []
        for i in range(num_bins):
            bin_elements = x[bin_indices == i]
            if len(bin_elements) > 0:
                weights = np.ones(len(bin_elements)) / len(bin_elements)
                sampled_elements = np.random.choice(bin_elements, expected_counts[i], replace=True, p=weights)
                y.extend(sampled_elements)

        # Step 5: Validasi Distribusi dengan `numpy.allclose()`
        y = np.array(y)
        actual_hist, _ = np.histogram(y, bins)
        actual_hist = actual_hist / np.sum(actual_hist)
        expected_hist = np.array(distr)
        tolerance = 1 / len(x)

        # Koreksi distribusi jika tidak sesuai
        max_attempts = 10
        for attempt in range(max_attempts):
            actual_hist, _ = np.histogram(y, bins)
            actual_hist = actual_hist / np.sum(actual_hist)

            if np.allclose(actual_hist, expected_hist, atol=tolerance, rtol=tolerance):
                break

            # Perbaiki jumlah elemen dalam setiap bin berdasarkan selisih dari expected_hist
            adjustments = np.round((expected_hist - actual_hist) * len(y)).astype(int)
            for i in range(num_bins):
                if adjustments[i] > 0:
                    extra_samples = np.random.choice(x[bin_indices == i], adjustments[i], replace=True)
                    y = np.append(y, extra_samples)
                elif adjustments[i] < 0:
                    y = np.delete(y, np.where(bin_indices == i)[0][:abs(adjustments[i])])

        # Step 6: Pastikan panjang `y` dalam batas yang diperbolehkan
        min_size = max(0, len(x) - len(distr))
        max_size = len(x) + len(distr)

        if len(y) < min_size:
            while len(y) < min_size:
                y = np.append(y, np.random.choice(x))
        elif len(y) > max_size:
            y = np.random.choice(y, max_size, replace=False)

        # Step 7: Acak urutan elemen di y
        np.random.shuffle(y)

        return y.tolist()
