# Magnetic door/window sensor

Magnetic Door Sensor
Features:
•Normally open reed switch
•ABS enclosure
•Rated current: 300 mA max
•Rated voltage: 100VDC
max
•Distance: 18 mm max
Magnet


Magnetic Door Sensor
 This sensor is essentially a reed switch,
encased in an ABS plastic shell.
 Normally the reed is 'open' (no
connection between the two wires). The
other half is a magnet.
 When the magnet is less than 18 mm
away, the reed switch closes. They're
often used to detect when a door or
window is opened.
 You can also pick up some double-sided
foam tape from a hardware store to
mount these, that works well without
needing screws.


Magnetic Door Sensor
 Door sensors have one reed switch and one magnet, creating a
closed circuit. If someone opens an armed door or window,
the magnet is pulled away from the switch, which breaks the circuit
and triggers an event.
 The reed switch is an electrical switch operated by an
applied magnetic field. It consists of a pair of contacts on
ferromagnetic metal reeds in a hermetically sealed glass
envelope.

# Motion Detector/PIR
The PIR sensor itself has two slots in it, each slot is made of
a special material that is sensitive to IR.
When the sensor is idle, both slots detect the same amount
of IR, the ambient amount radiated from the room or walls
or outdoors.
 When a warm body like a human or animal passes by, it
first intercepts one half of the PIR sensor, which causes
a positive differential change between the two halves.
 When the warm body leaves the sensing area, the reverse
happens, whereby the sensor generates a negative
differential change. These change pulses are what is
detected.

You need to have a Slide Switch to enable/disable the buzzer and LED alerts when the PIR is
activated.
Make the necessary changes in the connections to include the activation/deactivation of
the Intruder Alert System. (Use the TinkerCad Application on your Labtops to setup the
Intruder Alert System and submit it into the Learning Journal #6)

Let's explain the working principle. The module actually consists of a pyroelectric sensor, which generates energy when exposed to heat. That means when a human or animal body will get in the range of the sensor, it will detect a movement because the human or the animal body emits heat energy in a form of infrared radiation. That's where the name of the sensor comes from, a passive infrared sensor. And the term passive means that the sensor is not using any energy for detecting purposes, it just works by detecting the energy given off by the other objects. The module also consists a specially designed cover, named Frenzel lens, which focuses the infrared signals onto the pyroelectric sensor.

redwire is positive and black is negative

LED longer one is positive and shorter one is negative

# Electromechanical Relay - NO & NC contacts


Electromechanical relay consists of an electromagnet
which uses solenoid action to move a set of electrical
contacts from the open to the closed position or vice versa.
Figure below shows the construction of an electromechanical
relay.
Electromechanical (electronic)

NC : Normally Closed Contacts (1-2)
NC : Normally Closed Contacts are closed when relay is not energised.
NO : Normally Opened Contacts (1-4)
NO : Normally Opened Contacts are opened when relay is not energised.


When the Switch is OPENED, there is NO electricity supply to the Relay coil, therefore the coil is
not energised. The NO contact remains opened and thus the +5V supply cannot flow from the
COM contact to the NO contact. Thus the LED will not light up.

When the Switch is CLOSED, there is electricity
supply to the Relay coil, therefore the coil is
energised. The NO contact closed and thus the +5V
supply flows from the COM contact to the NO
contact. Thus the LED will light up.


1. When the Relay is activated (i.e. Alarm System =
Arm), power is supplied to the Alarm Circuit.
Depending on the state of the PIR sensor, the BUZZER
will turn ON or OFF.
2. When the PIR sensor is activated due to the
movement of infrared objects, the Vout will be 5 volt
which will energised the coil and the NO relay switch
will CLOSE and the BUZZER will turn ON.

# LED lighting circuit operation

# How an electromagnetic relay is used to control lights and buzzer

## Integration of the magnetic door, PIR, relay, buzzer with the WIFI Smart Switch to build an Intruder Alarm System.

## Connect Intruder Alarm circuit with electronic relays on a Breadboard using TinkerCad Application.

# Connection of electrical wires & components using Soldering and wire caps techniques.
