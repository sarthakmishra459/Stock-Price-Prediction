try:
    from main import app
    print("App initialized successfully!")
except Exception as e:
    print(f"Error initializing app: {e}")
    import traceback
    traceback.print_exc()
