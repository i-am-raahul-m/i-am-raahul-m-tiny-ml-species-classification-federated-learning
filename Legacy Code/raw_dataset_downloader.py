import requests
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

class iNaturalist_Downloader:
    # ID Mappings
    STATE_MAP = {"TN": 6842, "KL": 9627, "AP": 6810, "KA": 7043, "GJ": 6682, "RJ": 7215, "MP": 7745, "UP": 7468}
    TAXON_MAP = {"mammal": 40151, "reptile": 26036, "bird": 3, "amphibian": 20978}

    # HTTPS Request Essentials
    BASE_URL = "https://api.inaturalist.org/v1/observations"
    HEADERS = {
        "User-Agent": "inat-downloader/1.0"
    }
    PARAMS = {
        "taxon_id": 0,
        "place_id": 0,
        "quality_grade": "research",
        "photos": "true",
        "captive": "false",
        "rank": "species",
        "per_page": 200
    }

    # Directory Structure
    OUT_PATH = Path("D:/")

    # API Request Settings
    REQUEST_DELAY = 1.0
    MAX_PAGES = None

    # CONSTRUCTOR
    def __init__ (self, taxon:str, place:str, request_delay:float=1.0, max_pages:int=None):
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

        self.out_dir = self.OUT_PATH / f"{self.chosen_taxon}_{self.chosen_place}"
        self.img_dir = self.out_dir / "images"
        self.meta_dir = self.out_dir / "metadata"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)


    # UTILS 
    def _fetch_page(self, page):
        params = self.params.copy()
        params["page"] = page
        r = requests.get(self.BASE_URL, params=params, headers=self.HEADERS, timeout=30)
        r.raise_for_status()
        return r.json()


    def _download_image(self, url, path):
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)


    # MAIN
    def download_dataset(self):
        page = 1
        total_downloaded = 0

        print(f"Starting download... Taxon: {self.chosen_taxon}; Place: {self.chosen_place}.")

        while True:
            if self.MAX_PAGES and page > self.MAX_PAGES:
                break

            data = self._fetch_page(page)
            results = data.get("results", [])

            if not results:
                break

            print(f"Page {page} | Observations: {len(results)}")

            for obs in results:
                obs_id = obs["id"]

                # Save metadata
                meta_path = self.meta_dir / f"{obs_id}.json"

                if meta_path.exists():
                    print("Attempted to rewrite metadata.")
                    continue

                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(obs, f, ensure_ascii=False, indent=2)

                # Download and save images
                photos = obs.get("photos", [])
                for i, photo in enumerate(photos):
                    url = photo["url"].replace("square", "original")
                    img_name = f"{obs_id}_{i}.jpg"
                    img_path = self.img_dir / img_name

                    if img_path.exists():
                        continue

                    try:
                        self._download_image(url, img_path)
                        total_downloaded += 1
                    except Exception as e:
                        print(f"Failed image {img_name}: {e}")

            page += 1
            time.sleep(self.REQUEST_DELAY)

        print(f"\nDownload Completed. Pages visited: {page}; Images downloaded: {total_downloaded}.")


# THREAD RUNNER
def run_state_download(taxon, state_code):
    try:
        downloader = iNaturalist_Downloader(
            taxon=taxon,
            place=state_code,
        )
        downloader.download_dataset()
    except Exception as e:
        print(f"[ERROR] {state_code}: {e}")


if __name__ == "__main__":
    taxon = "mammal"

    states = list(iNaturalist_Downloader.STATE_MAP.keys())

    print(f"Starting parallel download for {len(states)} states...")

    with ThreadPoolExecutor(max_workers=len(states)) as executor:
        futures = [
            executor.submit(run_state_download, taxon, state)
            for state in states
        ]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print("Thread failed:", e)

    print("All downloads completed.")

