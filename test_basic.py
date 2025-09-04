"""Simple test to verify FedGATSage works"""

def test_imports():
    try:
        from src.community_detection import CommunityAwareProcessor
        from src.federated_learning import FedGATSageSystem
        print("✓ Core imports work")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_basic_creation():
    try:
        from src.community_detection import CommunityAwareProcessor
        processor = CommunityAwareProcessor()
        print("✓ Basic object creation works")
        return True
    except Exception as e:
        print(f"✗ Object creation failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing FedGATSage setup...")
    if test_imports() and test_basic_creation():
        print("✅ Setup works! Ready to use.")
    else:
        print("❌ Setup needs fixing.")