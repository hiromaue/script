import cv2
import pytesseract
from pytesseract import Output

# Tesseractの実行ファイルへのパスを設定（必要に応じて変更してください）
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'  # macOSの例
# pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'  # Windowsの例

def preprocess_image(image_path):
    # 画像をグレースケールで読み込む
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 解像度の向上のためリサイズ（必要に応じて）
    # img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    
    # 二値化
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # ノイズの除去
    img = cv2.medianBlur(img, 5)
    
    return img

def extract_text_from_image(preprocessed_image):
    # Tesseractの設定
    custom_config = r'--oem 3 --psm 6 -l jpn+eng'
    # 画像からテキストを抽出
    extracted_text = pytesseract.image_to_string(preprocessed_image, config=custom_config)
    
    return extracted_text

def main():
    image_path = ''  # 画像ファイルへのパス
    preprocessed_image = preprocess_image(image_path)
    extracted_text = extract_text_from_image(preprocessed_image)
    
    print("抽出されたテキスト:\n", extracted_text)

if __name__ == "__main__":
    main()
