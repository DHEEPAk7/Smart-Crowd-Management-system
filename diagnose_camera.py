WEBCAM DIAGNOSTICS SCRIPT
=========================

Run this script to test your webcam and identify the issue:

python -c "
import cv2
import sys

print('='*60)
print('WEBCAM DIAGNOSTIC TEST')
print('='*60)

# Test 1: Check available cameras
print('\n[TEST 1] Checking available cameras...')
available = []
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        available.append(i)
        print(f'  - Camera {i}: AVAILABLE')
        cap.release()
    else:
        print(f'  - Camera {i}: NOT available')

if not available:
    print('\nERROR: No cameras found!')
    print('Solution:')
    print('  1. Check USB cable is connected')
    print('  2. Check Device Manager > Imaging devices')
    print('  3. Restart computer')
    sys.exit(1)

print(f'\nFound {len(available)} camera(s): {available}')

# Test 2: Try to open first available camera
print('\n[TEST 2] Opening first camera...')
camera_idx = available[0]
cap = cv2.VideoCapture(camera_idx)

if not cap.isOpened():
    print(f'ERROR: Cannot open camera {camera_idx}')
    print('Solution:')
    print('  1. Check Windows Settings > Privacy & Security > Camera')
    print('  2. Check if other apps are using the camera')
    print('  3. Disable and re-enable camera in Device Manager')
    sys.exit(1)

print(f'Successfully opened camera {camera_idx}')

# Test 3: Try to read a frame
print('\n[TEST 3] Reading frame from camera...')
ret, frame = cap.read()

if not ret or frame is None:
    print('ERROR: Cannot read frame from camera')
    print('Solution:')
    print('  1. Check camera driver is up to date')
    print('  2. Try different USB port (back of computer)')
    print('  3. Restart the camera application')
    cap.release()
    sys.exit(1)

print(f'Successfully read frame: {frame.shape}')

# Test 4: Camera properties
print('\n[TEST 4] Camera properties:')
print(f'  - Width: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}')
print(f'  - Height: {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}')
print(f'  - FPS: {cap.get(cv2.CAP_PROP_FPS)}')
print(f'  - Brightness: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}')

# Test 5: Read 10 frames
print('\n[TEST 5] Reading 10 frames in sequence...')
success_count = 0
for i in range(10):
    ret, frame = cap.read()
    if ret:
        success_count += 1

print(f'Successfully read {success_count}/10 frames')

if success_count < 10:
    print('WARNING: Camera is dropping frames!')
    print('Solution: Try different USB port or camera')

cap.release()

print('\n' + '='*60)
print('ALL TESTS PASSED - Your camera is working!')
print('='*60)
"
