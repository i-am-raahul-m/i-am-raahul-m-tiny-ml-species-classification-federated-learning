import requests
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

class iNaturalist_Downloader:
    # ID Mappings
    STATE_MAP = {"TN": 6842, "KL": 9627, "AP": 6810, "KA": 7043, "GJ": 6682, "RJ": 7215, "MP": 7745, "UP": 7468}
    TAXON_MAP = {"mammal": 40151, "reptile": 26036, "bird": 3, "amphibian": 20978}

    # HTTPS Request Essentials
    BASE_URL = "https://api.inaturalist.org/v1/observations"
    HEADERS = {"User-Agent": "inat-downloader/1.0"}
    PARAMS = {
        "taxon_id": 0,
        "place_id": 0,
        "quality_grade": "research",
        "photos": "true",
        "captive": "false",
        "rank": "species",
        "per_page": 200
    }

    # Api Request Rate
    REQUEST_DELAY = 1.0

    def __init__(self, 
        taxon: str,
        place: str,
        out_dir: str = "D:/",
        img_per_obs: int = 2,
        request_delay: float = 1.0,
        max_pages: int = None,
    ):
        self.chosen_taxon = taxon.lower()
        self.chosen_place = place.upper()

        if self.chosen_taxon not in self.TAXON_MAP:
            raise ValueError("Unsupported taxon")
        if self.chosen_place not in self.STATE_MAP:
            raise ValueError("Unsupported state code")

        self.params = self.PARAMS.copy()
        self.params["taxon_id"] = self.TAXON_MAP[self.chosen_taxon]
        self.params["place_id"] = self.STATE_MAP[self.chosen_place]

        self.REQUEST_DELAY = request_delay
        self.MAX_PAGES = max_pages

        self.out_dir = Path(out_dir) / f"{self.chosen_taxon}_{self.chosen_place}"
        self.img_dir = self.out_dir / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        

        self.img_per_obs = img_per_obs

    # UTILS
    def _sanitize(self, name: str) -> str:
        return "".join(
            c if c.isalnum() or c in "_-" else "_"
            for c in name.strip().replace(" ", "_")
        )

    def _fetch_page(self, page: int):
        params = self.params.copy()
        params["page"] = page
        r = requests.get(self.BASE_URL, params=params, headers=self.HEADERS, timeout=30)

        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            time.sleep(5)
            return {"results": []}
        
        return r.json()

    def _download_image(self, url: str, path: str):
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)

    # MAIN
    def download_dataset(self):
        page = 1
        total_downloaded = 0

        print(f"[{self.chosen_place}] Starting download for {self.chosen_taxon}")

        while True:
            if self.MAX_PAGES and page > self.MAX_PAGES:
                break

            data = self._fetch_page(page)
            results = data.get("results", [])
            if not results:
                break

            print(f"[{self.chosen_place}] Page {page} | Observations: {len(results)}")

            for obs in results:
                obs_id = obs["id"]
                taxon = obs.get("taxon", {})

                latin_name = taxon.get("name", "unknown")
                common_name = (
                    taxon.get("preferred_common_name")
                    or taxon.get("english_common_name")
                    or "unknown"
                )

                latin_name = self._sanitize(latin_name)
                common_name = self._sanitize(common_name)

                species_dir = self.img_dir / f"f{common_name}-{latin_name}"
                species_dir.mkdir(parents=True, exist_ok=True)

                photos = obs.get("photos", [])[:self.img_per_obs]

                for i, photo in enumerate(photos):
                    url = photo["url"].replace("square", "original")
                    img_path = species_dir / f"{obs_id}_{i}.jpg"

                    if img_path.exists():
                        continue
                    
                    # Retry to download an image upto 3 times
                    for _ in range(3):
                        try:
                            self._download_image(url, img_path)
                            total_downloaded += 1
                            break
                        except Exception as e:
                            print(f"[{self.chosen_place}] Failed image {img_path.name}: {e}")
                            time.sleep(1)
                            
            page += 1
            time.sleep(self.REQUEST_DELAY)

        print(f"[{self.chosen_place}] Done. Images downloaded: {total_downloaded}")


# THREAD RUNNER
def run_state_download(taxon, state_code):
    try:
        downloader = iNaturalist_Downloader(taxon=taxon, place=state_code, request_delay=1.5)
        downloader.download_dataset()
    except Exception as e:
        print(f"[ERROR] {state_code}: {e}")


if __name__ == "__main__":
    taxon = "mammal"
    states = list(iNaturalist_Downloader.STATE_MAP.keys())

    print(f"Starting parallel download for {len(states)} states...")

    with ThreadPoolExecutor(max_workers=len(states)) as executor:
        futures = [executor.submit(run_state_download, taxon, state) for state in states]

        for future in as_completed(futures):
            future.result()

    print("All downloads completed.")
