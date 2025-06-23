# Feature Request: Key-up Triggered Combos for ZMK

Hello ZMK community! I've been thinking about ways to enhance our combo system, and I'd love to share an idea that could potentially improve combo accuracy and flexibility. I'm excited to hear your thoughts and suggestions on this proposal.

## Current Situation and Challenge

Currently, ZMK combos are triggered when the specified keys are pressed down within the `timeout` window. While this works well in many cases, there are some issues:

1. **Mispresses**: Fast typist will find a lot of mistriggered combos during normal typing. Typists will also fail to trigger combos if their combo press was a little sloppy.
1. **Over-reliance on timing**: Similar to traditional Homerow-mods, timing becomes the only factor, which is unpredictable and challenging to keep track during normal typing.

## Proposed Enhancement: Key-up Triggered Combos

What if we could calculate combos when keys are released instead of when they're pressed? The system would wait for the key releases for combo-mapped keys when the key presses looks like a combo is about to happen to calculate.

1. Increased accuracy: By waiting for key releases, we reduce the chance of accidental triggers during fast typing. Fast typist can roll words while still having combos on easy to reach places.
2. Comfort: Arguably it is much more comfortable to roll/string and hold the keys and release them together at once, than trying to hit them all at once the first try.
2. More combo possibilities: This could make longer key combinations more viable, since you don't have to get all the keys in one go, just press required keys and release together.

## Key-up Combo System: Explanation

### Key Parameters
- timeout: 150 ms
- release-timeout: 100ms
- invalidate_release_size: 2
- combo: 3 keys (for this example)

### Process

1. **Initial Key Press**
   When a user presses a key associated with a combo, the system initiates a 150ms timeout window.
   
   - If the user holds this key for more than 100ms without pressing any other keys:
     - If a hold-tap behavior is defined, it takes precedence.
     - If no hold-tap is defined, the system sends the default key-hold (e.g., key repeat).
     - If no default key-hold is defined, the system continues waiting for the combo until key release.
   
   - If the user releases the key and no other keys were pressed, the default keypress is sent.

   *Note: Hold-tap behavior only triggers if a single key is held beyond the timeout with no other keys pressed. If multiple keys are held down, hold-tap does not activate.*

2. **Additional Key Presses**
   The user may press more keys, keeping each key held down. The system continues to monitor these presses.

   *Note: If a user presses a key not part of the combo during this process, this extra key is ignored and does not affect the combo execution.*

   *Note: Combo keys are defined by key positions, not the characters they produce. This means a combo can potentially span across layers.*

3. **Key Release Phase**
   As the user starts to release keys, different scenarios can occur:
   
   - If only one key is released, the default keypress for that key is sent. This is because the invalidate_release_size is set to 2 keys.
   
   - If two keys are released and 100ms elapses since the latest key release, no action is taken. This is because the invalidate_release_size equals 2 keys.
   
   - If all three keys are released within the 100ms release-timeout, the combo is activated and sent.

   *Note: If a user's releases span just beyond the 100ms release-timeout, the combo will be missed. The system will send individual keypresses or nothing instead depending on `invalidate_release_size`. However, this scenario is less likely to occur compared to timing issues with key-down combos, and the user won't need to delete unwanted characters if `invalidate_release_size` is 2 or more.*

4. **Timeout Behavior**
   The 150ms timeout is a window during which the system waits for additional key presses. No timeout-related action occurs as long as multiple keys are held down together, or when the last key is released.

### Possible Configuration Options

To make this feature flexible and powerful, we could consider options like:

- Activation mode: Choose between key-down (current behavior) and key-up triggering.
- Timeout window: Set how long to wait for additional key presses after the first key-up event.

## Community Input Wanted!

I'm really excited about the potential of this idea, but I know there's a lot to consider. I'd love to hear your thoughts.

Let's discuss and refine this idea together.
