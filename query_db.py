import cv2
import numpy as np
from insightface.app import FaceAnalysis
from pinecone import Pinecone

# ---- Init Pinecone ---- #
pc = Pinecone(api_key="pcsk_4Larmy_EmaoUbmwSGpwH8ree9h8zKP8kAZyhvAQGbViTtjergsFXQPdoYjV4KJGuGiYPhU")
index = pc.Index("face-recognition")

# ---- Init InsightFace ---- #
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])  # use GPU if you want: ["CUDAExecutionProvider"]
app.prepare(ctx_id=0)

# ---- Load Image ---- #
img_path = "/Users/zikangjiang/face_search_app/test4.png"
img = cv2.imread(img_path)

# ---- Detect Faces ---- #
faces = app.get(img)

# ---- Process Each Face ---- #
for face in faces:
    # Draw bounding box
    box = face.bbox.astype(int)
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # Draw facial landmarks
    # for (x, y) in face.kps.astype(int):
    #     cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

    # Extract embedding
    embedding = face.normed_embedding.tolist()

    # Query Pinecone for top 2 matches
    results = index.query(
        vector=embedding,
        top_k=2,
        include_metadata=True,
        include_values=False
    )

    # Annotate with best match
    top_matches = results.get("matches", [])
    if top_matches:
        name1 = top_matches[0]["metadata"].get("name", "Unknown")
        score1 = top_matches[0]["score"]
        name2 = top_matches[1]["metadata"].get("name", "Unknown") if len(top_matches) > 1 else "None"
        score2 = top_matches[1]["score"] if len(top_matches) > 1 else 0.0

        label = f"{name1} ({score1:.2f}) | {name2} ({score2:.2f})"
    else:
        label = "No match"

    # Put label above bounding box
    def draw_text_with_background(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, text_color=(255,255,255), bg_color=(0,0,0), thickness=2):
        """Draws text with background rectangle for readability."""
        x, y = position
        (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(img, (x, y - h - 6), (x + w + 4, y + 4), bg_color, -1)
        cv2.putText(img, text, (x + 2, y - 2), font, font_scale, text_color, thickness, cv2.LINE_AA)

    draw_text_with_background(img, label, (box[0], box[1] - 10))

# ---- Show Results ---- #
cv2.imshow("Face Recognition Results", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
