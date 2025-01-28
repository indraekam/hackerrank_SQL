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

        # Konversi ke numpy array
        x = np.array(x)

        # Step 1: Hapus Outliers (Data harus berada dalam 2 standar deviasi)
        mean_x, std_x = np.mean(x), np.std(x)
        x = x[np.abs(x - mean_x) <= 2 * std_x]

        # Step 2: Buat bins sesuai distribusi `distr`
        num_bins = len(distr)
        bins = np.linspace(np.min(x), np.max(x), num_bins + 1)
        bin_indices = np.digitize(x, bins) - 1  # Mengelompokkan data ke dalam bins

        # Step 3: Hitung jumlah elemen yang harus diambil dari setiap bin
        total_count = len(x)
        expected_counts = np.round(distr * total_count).astype(int)

        # Step 4: Sampling berdasarkan distribusi `distr`
        y = np.hstack([
            np.random.choice(x[bin_indices == i], expected_counts[i], replace=True)
            if np.any(bin_indices == i) else []
            for i in range(num_bins)
        ])

        # Step 5: Koreksi panjang `y` jika terlalu kecil atau terlalu besar
        min_size = max(0, len(x) - len(distr))
        max_size = len(x) + len(distr)
        y = np.random.choice(y, np.clip(len(y), min_size, max_size), replace=False)

        # Step 6: Acak hasil sebelum return
        np.random.shuffle(y)

        return y.tolist()
