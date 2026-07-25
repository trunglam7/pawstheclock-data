import json
import requests
from bs4 import BeautifulSoup
from ollama import chat
from ollama import ChatResponse
from urllib.parse import urljoin
import re
from pydantic import BaseModel, Field
from typing import List
import instructor

# 1. Define the schema structure using Pydantic
class AnimalRecord(BaseModel):
    name: str = Field(description="The name of the animal")
    breed: str = Field(description="The breed of the animal")
    comments: str = Field(description="The full comments block including status, warnings, deadlines, and days out")
    id: str = Field(description="The unique animal tracking ID or code exactly as it appears in the text block")
    profile_link: str = Field(description="The absolute profile link URL extracted for this animal")
    image_link: str = Field(description="The absolute image URL link. If missing, capture it as an empty string.")

# 2. Patch your Ollama client using instructor
# For llama3.1, instructor will automatically utilize native tool/JSON parsing modes
client = instructor.from_provider("ollama/llama3.1:8b", mode=instructor.Mode.JSON)

def fetch_and_clean_shelter_data(url):
    print(f"🌍 [UNIVERSAL] Scanning target: {url}", flush=True)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Failed to reach site: {e}")
        return ""
        
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Strip heavy structural components that never contain individual animal profiles
    for noisy_tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form", "svg"]):
        noisy_tag.decompose()

    final_records = []
    seen_images = set()

    # Step 1: Identify all potential animal images on the page
    all_imgs = soup.find_all('img', src=True)
    
    for img in all_imgs:
        img_src = urljoin(url, img['src'])
        
        # Skip small icons, layout spacers, tracking pixels, and decorative graphics
        if any(bad in img_src.lower() for bad in ['spacer', 'loader', 'gif', 'logo', 'icon', 'theme', 'avatar']):
            continue
        if img_src in seen_images:
            continue
            
        seen_images.add(img_src)

        # Step 2: Climb up the DOM tree to find the animal's contextual wrapper
        # We start looking at the parent, grandparent, etc., up to 4 levels high
        parent_box = None
        current_element = img
        
        for _ in range(4):
            current_element = current_element.parent
            if not current_element:
                break
            
            # If the block contains text commonly associated with a listing, it's our container
            box_text = current_element.get_text(" ", strip=True)
            if any(kwd in box_text.lower() for kwd in ["id:", "id#", "breed", "age", "gender", "adopt", "name"]):
                parent_box = current_element
                break # Found the closest container holding both image and data
        
        # Fallback: If no smart text container was found, evaluate the immediate parent element
        if not parent_box:
            parent_box = img.parent

        # Step 3: Extract the Profile Link using proximity
        profile_link = "No Link Found"
        # Search inside our smart container first
        inner_a = parent_box.find('a', href=True) if parent_box else None
        
        if inner_a:
            profile_link = urljoin(url, inner_a['href'])
        else:
            # If it's a split column layout, look horizontally at adjacent elements
            sibling = img.find_next('a', href=True) or img.find_previous('a', href=True)
            if sibling:
                profile_link = urljoin(url, sibling['href'])

        # Step 4: Extract the Text block using proximity
        text_details = ""
        if parent_box:
            # Gather text, separating tags with clean spaces
            text_details = parent_box.get_text(" ", strip=True)
        else:
            # Fallback to the nearest text block string
            text_details = img.find_next(text=True) or "No details extracted"

        # Clean up excess whitespace strings
        text_details = " ".join(text_details.split())
        
        # Optional: Limit block size if it accidentally grabbed the entire webpage container
        if len(text_details) > 400:
            text_details = text_details[:400] + "..."

        # Step 5: Format the record string
        record = (
            f"<ANIMAL_RECORD>\n"
            f"[PROFILE_LINK]: {profile_link}\n"
            f"[IMAGE_LINK]: {img_src}\n"
            f"[DETAILS]: {text_details}\n"
            f"</ANIMAL_RECORD>"
        )
        final_records.append(record)

    return "\n\n".join(final_records)

target_url = "https://24petconnect.com/RVSDPublic" 
cleaned_html_text = fetch_and_clean_shelter_data(target_url)

if cleaned_html_text:
    print(f"\n🤖 [STAGE 2] Extracting high-urgency animals via Instructor...", flush=True)
    
    try:
        # Without stream=True, this blocks until the entire list is fully built and validated
        extracted_animals = client.chat.completions.create(
            model='llama3.1:8b',
            response_model=List[AnimalRecord],
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are a strict data extraction utility. Isolate and parse every <ANIMAL_RECORD> block accurately.\n\n"
                        "FILTERING FOR URGENCY: Do not extract any animal that is safely out of the physical shelter environment. If the record indicates the animal is 'In a Foster Home', 'Adopted', 'Transferred', or otherwise out of the facility, skip it entirely. We are only extracting animals physically at the shelter who still face risk."
                    )
                },
                {
                    "role": "user", 
                    "content": f"Shelter Data Stream: {cleaned_html_text}"
                }
            ],
            timeout=None
        )

        # Convert the verified Pydantic objects directly into standard dictionaries
        all_animals = [animal.model_dump() for animal in extracted_animals]

        # Save to JSON file
        output_filename = "shelter_animals.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(all_animals, f, indent=4, ensure_ascii=False)
            
        print(f"✅ Extraction complete! Found {len(all_animals)} high-risk animals.")
        print(f"📁 Data cleanly exported to '{output_filename}'")

    except Exception as e:
        print(f"\n❌ Instructor/Ollama extraction failed: {e}")