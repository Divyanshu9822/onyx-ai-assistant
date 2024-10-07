from dotenv import load_dotenv

load_dotenv()

import uuid
import cloudinary
from cloudinary import CloudinaryImage
import cloudinary.uploader
import cloudinary.api


config = cloudinary.config(secure=True)


def upload_image(base64_image: str) -> str:
    random_id = str(uuid.uuid4())

    base64_image_with_prefix = f"data:image/jpeg;base64,{base64_image}"

    response = cloudinary.uploader.upload(
        base64_image_with_prefix,
        public_id=random_id,
        unique_filename=False,
        overwrite=True,
    )

    srcURL = CloudinaryImage(random_id).build_url()
    return srcURL
