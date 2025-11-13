import sys
sys.path.append('src')
from data_preprocessing import DataPreprocessor

print("="*60)
print("TESTING TEXT CLEANING")
print("="*60)

preprocessor = DataPreprocessor()

test_texts = [
    "Women are not smart enough for leadership",
    "That ethnic group does not belong here",
    "People from that country are all the same",
    "She only got that job because she is a woman",
]

print("\n")
for i, text in enumerate(test_texts, 1):
    cleaned = preprocessor.clean_text(text)
    print(f"Test {i}:")
    print(f"  Original: {text}")
    print(f"  Cleaned:  {cleaned}")
    
    # Check if critical words are preserved
    if "not" in text.lower():
        if "not" in cleaned:
            print(f"  Status:   ✓ 'not' preserved")
        else:
            print(f"  Status:   ✗ ERROR: 'not' was removed!")
    
    if "are" in text.lower():
        if "are" in cleaned:
            print(f"  Status:   ✓ 'are' preserved")
        else:
            print(f"  Status:   ✗ ERROR: 'are' was removed!")
    
    if "does" in text.lower():
        if "does" in cleaned:
            print(f"  Status:   ✓ 'does' preserved")
        else:
            print(f"  Status:   ✗ ERROR: 'does' was removed!")
    
    print()

print("="*60)
print("If you see any ERROR messages above, the wrong file is loaded!")
print("="*60)