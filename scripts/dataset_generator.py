import os
import hashlib
from pathlib import Path
from PIL import Image
import imagehash
from tqdm import tqdm
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# ══════════════════════════════════════════════════════════════
# 1. EDIT THESE BEFORE RUNNING
# ══════════════════════════════════════════════════════════════

KEYWORDS = [
    'ISKCON Temple Ahmedabad Satellite',
    'Shree Swaminarayan Mandir Kalupur Ahmedabad',
    'BAPS Shri Swaminarayan Mandir Shahibaug Ahmedabad',
    'BAPS Shri Swaminarayan Mandir Devjipura Satellite Ahmedabad',
    'Shree Balaji Temple Chharodi Sarkhej Ahmedabad',
    'Shri Maa Vaishnodevi Temple Ahmedabad NH147',
    'Shree Camp Hanumanji Mandir Airport Road Ahmedabad',
    'Shri Jagannathji Mandir Jamalpur Ahmedabad',
    'Nagardevi Shri Bhadrakali Mata Temple Bhadra Fort Ahmedabad',
    'Shree Samartheshwar Mahadev Temple Ellisbridge Ahmedabad',
    'Koteshwar Mahadev Temple north Ahmedabad',
    'Lambha Baliyadev Temple Laxmipura Ahmedabad',
    'Shree Mahalaxmi Temple Raikhad Ahmedabad',
    'Shri Devendrashwar Mahadev Temple Drive-In Road Memnagar Ahmedabad',
    'Vrajdham Haveli Temple Satellite Jodhpur Village Ahmedabad',
    'Shree Omkareshwar Mahadev Temple Jodhpur Village Ahmedabad',
    'Shree Chamunda Mata Mandir Raikhad Ahmedabad',
    'Shree Kal Bhairav Mandir Raikhad Ahmedabad',
    'Shree Gorakhnath Mahadev Khadia Ahmedabad',
    'Shree Ramji Mandir Manek Chowk Ahmedabad',
    'Shree Shani Dev Mandir Dariyapur Ahmedabad',
    'Shree Sheetla Mata Mandir Shahpur Ahmedabad',
    'Shree Shankar Bhagwan Mahadev Maninagar Ahmedabad',
    'Shree Gayatri Mandir Maninagar Ahmedabad',
    'Shree Hatkeshwar Mahadev Khokhra Ahmedabad',
    'Shree Meldi Mata Mandir Khokhra Hatkeshwar Ahmedabad',
    'Shree Umiya Mata Mandir Nikol Vastral Ahmedabad',
    'Shree Shani Dev Mandir Nikol Vastral Ahmedabad',
    'Shree Ramji Mandir Paldi Ahmedabad',
    'Shree Rameshwar Mahadev Ambawadi Ahmedabad',
    'Shree Hanuman Mandir Navrangpura Ahmedabad',
    'Shree Nilkanth Mahadev Naranpura Ahmedabad',
    'Shree Bhidbhanjan Hanuman Mandir Ghatlodia Ahmedabad',
    'Shree Mahakali Mata Mandir Sola Ahmedabad',
    'Shree Siddheshwar Mahadev Sabarmati Ahmedabad',
    'Shree Ashapura Mata Mandir Motera Ahmedabad',
    'Shree Radhakrishna Mandir Chandkheda Ahmedabad',
    'Shree Hanuman Mandir Chandkheda Ahmedabad',
    'Shree Khodiyar Mata Mandir Behrampura Ahmedabad',
    'Shree Chamunda Mata Mandir Danilimda Ahmedabad',
    'Shree Jogani Mata Mandir Isanpur south Ahmedabad',
    'Shree Somnath Mahadev Vatva south Ahmedabad',
    'Shree Shiv Shakti Mandir Bopal west Ahmedabad',
    'Shree Khodiyar Mata Mandir Ghuma west Ahmedabad',
    'Shree Swami Narayan Mandir South Bopal west Ahmedabad',
    'Shree Hanuman Mandir Shela west Ahmedabad',
    'Shree Khodiyar Mata Mandir Gota west Ahmedabad',
    'Shree Hanuman Mandir Gota west Ahmedabad',
    'Shree Nilkanth Mahadev Mandir Gota west Ahmedabad',
    'Shree Gayatri Mandir Gota west Ahmedabad',
    'Shree Swaminarayan Mandir Gota west Ahmedabad',
    'Shri Maa Vaishnodevi Temple Vaishnodevi Circle north-west Ahmedabad',
    'Shree Hanuman Mandir Vaishnodevi Circle north-west Ahmedabad',
    'Shree Shankar Mahadev Mandir SG Highway near Vaishnodevi north-west Ahmedabad',
    'Shree Ashapura Mata Mandir Vaishnodevi Circle north-west Ahmedabad',
    'Shree Jogani Mata Mandir Vaishnodevi Circle north-west Ahmedabad',
    'Shree Khodiyar Mata Mandir Godasar south-east Ahmedabad',
    'Shree Hanuman Mandir Godasar south-east Ahmedabad',
    'Shree Nilkanth Mahadev Mandir Godasar south-east Ahmedabad',
    'Shree Jogani Mata Mandir Godasar south-east Ahmedabad',
    'Shree Shiv Mandir Ambli west Ahmedabad',
    'Shree Hanuman Mandir Ambli west Ahmedabad',
    'Shree Ashapura Mata Mandir Ambli west Ahmedabad',
    'Shree Shankar Mahadev Mandir Sindhu Bhavan Road west Ahmedabad',
    'Shree Hanuman Mandir Sindhu Bhavan Road west Ahmedabad',
    'Shree Radhavallabh Mandir Thaltej west Ahmedabad',
    'Shree Hanuman Mandir Thaltej west Ahmedabad',
    'Shree Nilkanth Mahadev Mandir Thaltej west Ahmedabad',
    'Shree Khodiyar Mata Mandir Thaltej west Ahmedabad',
    'Shree Hanuman Mandir Bopal west Ahmedabad',
    'Shree Swaminarayan Mandir Bopal west Ahmedabad',
    'Shree Gayatri Mandir Bopal west Ahmedabad',
    'Shree Hanuman Mandir South Bopal west Ahmedabad',
    'Shree Nilkanth Mahadev Mandir Ghuma west Ahmedabad',
    'Shree Vejalpur Hanuman Mandir Vejalpur west Ahmedabad',
    'Shree Nilkanth Mahadev Mandir Vejalpur west Ahmedabad',
    'Shree Khodiyar Mata Mandir Vejalpur west Ahmedabad',
    'Shree Gayatri Mandir Vejalpur west Ahmedabad',
    'Shree Ashapura Mata Mandir Makarba south-west Ahmedabad',
    'Shree Hanuman Mandir Makarba south-west Ahmedabad',
    'Shree Shankar Mahadev Mandir Makarba south-west Ahmedabad',
    'Shree Ramdev Pir Mandir Makarba south-west Ahmedabad',
]

# Caption prefix — concept token for this LoRA
CAPTION_PREFIX = 'a photo of sks temple,'

# Images to download per keyword (Google + Bing each)
IMAGES_PER_KEYWORD = 1

# Hard cap: only keep this many images after filtering (first run = 30)
MAX_TOTAL_IMAGES = 60

# ══════════════════════════════════════════════════════════════
# 2. PATHS
# ══════════════════════════════════════════════════════════════

RAW_DIR   = Path(r'C:\Users\sarja\OneDrive\Desktop\lora_project\dataset\raw')
FINAL_DIR = Path(r'C:\Users\sarja\OneDrive\Desktop\lora_project\dataset\final')
RAW_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 3. STEP 1 — SCRAPE IMAGES
# ══════════════════════════════════════════════════════════════

def scrape_images():
    print('\n[Step 1] Scraping images...')
    for keyword in KEYWORDS:
        safe_kw = keyword.replace(' ', '_').replace('/', '_')[:60]
        kw_dir = RAW_DIR / safe_kw
        kw_dir.mkdir(parents=True, exist_ok=True)

        # Google
        g_dir = kw_dir / 'google'
        g_dir.mkdir(exist_ok=True)
        try:
            google_crawler = GoogleImageCrawler(storage={'root_dir': str(g_dir)})
            google_crawler.crawl(keyword=keyword, max_num=IMAGES_PER_KEYWORD)
        except Exception as e:
            print(f'  Google scrape failed for "{keyword}": {e}')

        # Bing
        b_dir = kw_dir / 'bing'
        b_dir.mkdir(exist_ok=True)
        try:
            bing_crawler = BingImageCrawler(storage={'root_dir': str(b_dir)})
            bing_crawler.crawl(keyword=keyword, max_num=IMAGES_PER_KEYWORD)
        except Exception as e:
            print(f'  Bing scrape failed for "{keyword}": {e}')

    print('[Step 1] Scraping complete.')

# ══════════════════════════════════════════════════════════════
# 4. STEP 2 — FILTER + DEDUPLICATE + RESIZE
# ══════════════════════════════════════════════════════════════

MIN_SIZE = 256  # minimum pixel dimension

def is_valid_image(path):
    try:
        img = Image.open(path)
        img.verify()
        img = Image.open(path)
        w, h = img.size
        return w >= MIN_SIZE and h >= MIN_SIZE
    except Exception:
        return False

def collect_all_raw():
    paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp'):
        paths.extend(RAW_DIR.rglob(ext))
    return paths

def filter_deduplicate_resize():
    print('\n[Step 2] Filtering, deduplicating, and resizing...')
    all_paths = collect_all_raw()
    print(f'  Found {len(all_paths)} raw images.')

    seen_hashes = set()
    valid_images = []

    for p in tqdm(all_paths, desc='  Filtering'):
        if len(valid_images) >= MAX_TOTAL_IMAGES:
            break
        if not is_valid_image(p):
            continue
        try:
            img = Image.open(p).convert('RGB')
            h = str(imagehash.phash(img))
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            valid_images.append(img)
        except Exception:
            continue

    print(f'  {len(valid_images)} images after filtering and deduplication (cap: {MAX_TOTAL_IMAGES}).')

    for i, img in enumerate(tqdm(valid_images, desc='  Resizing')):
        # Smart center-crop to square then resize to 1024x1024
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top  = (h - side) // 2
        img  = img.crop((left, top, left + side, top + side))
        img  = img.resize((1024, 1024), Image.LANCZOS)
        out_path = FINAL_DIR / f'image_{i+1:04d}.jpg'
        img.save(out_path, 'JPEG', quality=95)

    print(f'[Step 2] Done. {len(valid_images)} images saved to {FINAL_DIR}')
    return len(valid_images)

# ══════════════════════════════════════════════════════════════
# 5. STEP 3 — CAPTION WITH BLIP
# ══════════════════════════════════════════════════════════════

def generate_captions():
    print('\n[Step 3] Generating captions with BLIP...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'  Using device: {device}')

    processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-large')
    model = BlipForConditionalGeneration.from_pretrained(
        'Salesforce/blip-image-captioning-large').to(device)

    image_files = sorted(FINAL_DIR.glob('*.jpg'))
    for img_path in tqdm(image_files, desc='  Captioning'):
        txt_path = img_path.with_suffix('.txt')
        if txt_path.exists():
            continue  # skip already-captioned
        try:
            img = Image.open(img_path).convert('RGB')
            inputs = processor(img, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(out[0], skip_special_tokens=True)
            full_caption = f'{CAPTION_PREFIX} {caption}'
            txt_path.write_text(full_caption, encoding='utf-8')
        except Exception as e:
            print(f'  Warning: could not caption {img_path.name}: {e}')

    print('[Step 3] Captioning complete.')

# ══════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    scrape_images()
    count = filter_deduplicate_resize()
    generate_captions()
    print(f'\n✅ Dataset ready! {count} images in {FINAL_DIR}')
    print('Each image has a matching .txt caption file.')
