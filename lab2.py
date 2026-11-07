# ==================================================
# TASK 1: PEAS SPECIFICATIONS
# ==================================================

def get_agent_specifications():

    delivery_robot = [

        (
            "Performance measure",
            [
                "Deliver the correct parcel to the correct destination",
                "Meet delivery-time targets",
                "Avoid collisions and parcel damage",
                "Minimize energy use",
            ]
        ),

        (
            "Environment",
            [
                "Sidewalks and building corridors",
                "Pedestrians and other robots",
                "Curbs, doors, ramps, and elevators",
                "Parcel pickup and drop-off points",
                "Changing weather and lighting",
            ]
        ),

        (
            "Actuators",
            [
                "Drive motors and steering",
                "Brakes",
                "Robotic arm or parcel latch",
                "Status display, speaker, or warning light",
            ]
        ),

        (
            "Sensors",
            [
                "RGB or depth cameras",
                "Lidar or ultrasonic range sensors",
                "Wheel encoders and inertial sensors",
                "GPS outdoors",
                "Parcel-latch and battery sensors",
            ]
        ),

        (
            "Constraint or trade-off",
            [
                "Moving faster can improve delivery time but increases "
                "stopping distance and collision risk."
            ]
        ),

    ]

    student_support = [

        (
            "Performance measure",
            [
                "Give accurate answers grounded in approved course information",
                "Respond in a useful amount of time",
                "Protect student privacy",
                "Escalate uncertain or sensitive cases",
            ]
        ),

        (
            "Environment",
            [
                "Student chat interface",
                "Course and university support questions",
                "Approved FAQ and policy documents",
                "Support staff and ticket queue",
                "Different student language and accessibility needs",
            ]
        ),

        (
            "Actuators",
            [
                "Send a chat reply",
                "Ask a clarifying question",
                "Create an escalation ticket",
                "Provide an approved document link",
            ]
        ),

        (
            "Sensors and inputs",
            [
                "Student messages and conversation context",
                "Read-only search of approved FAQs",
                "Read-only course catalog and policy search",
                "Status of an existing support ticket",
            ]
        ),

        (
            "Operational assumptions",
            [
                "The agent can read the current conversation.",

                "It can search approved FAQ, course catalog, "
                "and policy pages.",

                "It can inspect existing ticket status and "
                "create an escalation ticket.",

                "It cannot access grades, private student "
                "records, or email accounts.",

                "It cannot change official records or "
                "make enrollment decisions.",
            ]
        ),

        (
            "Constraint or trade-off",
            [
                "Searching trusted sources and escalating uncertain "
                "answers can improve accuracy but may increase "
                "response time."
            ]
        ),

    ]

    return [

        ("Delivery robot", delivery_robot),

        ("LLM-based student-support agent", student_support),

    ]


def display_section(title, entries):

    print(f"\n{title}:\n")

    for entry in entries:

        print(f"  - {entry}")


def display_peas():

    print("\nTASK 1: PEAS SPECIFICATIONS\n")

    for agent_name, sections in get_agent_specifications():

        print("=" * 70)

        print(f"\nAgent: {agent_name}")

        for title, entries in sections:

            display_section(title, entries)

        print()


# ==================================================
# TASK 2: ENVIRONMENT CLASSIFICATIONS
# ==================================================

def classify_environment(complete_state, fixed_outcomes):

    observability = (
        "Fully observable"
        if complete_state
        else "Partially observable"
    )

    dynamics = (
        "Deterministic"
        if fixed_outcomes
        else "Stochastic"
    )

    return observability, dynamics


def get_environment_cases():

    # Format:
    # (
    #     system,
    #     complete state available,
    #     fixed action outcomes,
    #     observable example,
    #     hidden example,
    #     action outcome
    # )

    return [

        (
            "Delivery robot — real world",

            False,

            False,

            "Camera and lidar readings",

            "A pedestrian's intent or an occluded obstacle",

            "A drive command may have different outcomes due to "
            "traction, moving people, or sensor noise."
        ),

        (
            "Student-support agent — real service",

            False,

            False,

            "Student message and search results",

            "Unstated student circumstances or a policy change "
            "not yet indexed",

            "A reply may resolve the question, prompt "
            "clarification, or require escalation."
        ),

        (
            "Delivery robot — simplified simulator",

            True,

            True,

            "Assumption: the simulator exposes all modeled "
            "positions, obstacles, battery, and parcel state.",

            "None in this simplified simulator",

            "Assumption: fixed movement rules map each command "
            "to one result."
        ),

        (
            "Student-support agent — simplified simulator",

            True,

            True,

            "Assumption: the simulator provides the complete "
            "student/query state and a fixed knowledge base.",

            "None in this simplified simulator",

            "Assumption: each agent action maps to a fixed outcome."
        ),

    ]


def display_environment_classifications():

    print("\nTASK 2: ENVIRONMENT CLASSIFICATIONS\n")

    for case in get_environment_cases():

        (
            system,
            complete_state,
            fixed_outcomes,
            observable,
            hidden,
            outcome
        ) = case

        observability, dynamics = classify_environment(
            complete_state,
            fixed_outcomes
        )

        print("=" * 70)

        print(f"\nSystem: {system}")

        details = [

            ("Observability", observability),

            ("Dynamics", dynamics),

            ("Observable example", observable),

            ("Hidden example", hidden),

            ("Action outcome", outcome),

        ]

        for label, value in details:

            print(f"\n  {label}: {value}")

        print()


# ==================================================
# TASK 3: ROOM CONTROLLER WITH MEMORY
# ==================================================

class RoomController:

    def __init__(self):

        self.previous_target_mode = None

        self.rules = [

            (
                "OFF",
                lambda temperature, occupied: not occupied
            ),

            (
                "HEAT",
                lambda temperature, occupied: temperature < 18
            ),

            (
                "COOL",
                lambda temperature, occupied: temperature > 26
            ),

        ]

    def select_target(self, temperature_c, occupied):

        return next(

            (
                mode

                for mode, condition in self.rules

                if condition(temperature_c, occupied)

            ),

            "IDLE"

        )

    def update(self, temperature_c, occupied):

        old_mode = self.previous_target_mode

        new_mode = self.select_target(
            temperature_c,
            occupied
        )

        command_required = old_mode != new_mode

        self.previous_target_mode = new_mode

        return {

            "previous_mode": old_mode,

            "target_mode": new_mode,

            "send_command": command_required,

        }


def run_room_controller():

    test_cases = [

        ("Initial heat", 17, True),

        ("Repeated heat percept", 17, True),

        ("Lower exact boundary", 18, True),

        ("Upper exact boundary", 26, True),

        ("Mode transition to cool", 27, True),

        ("Repeated cool percept", 27, True),

        ("Empty room priority", 30, False),

    ]

    controller = RoomController()

    commands_sent = 0

    print("\nTASK 3: ROOM-CONTROLLER CASES\n")

    for number, case in enumerate(test_cases, start=1):

        name, temperature, occupied = case

        result = controller.update(
            temperature,
            occupied
        )

        commands_sent += int(result["send_command"])

        command_result = (
            "SEND"
            if result["send_command"]
            else "NO COMMAND"
        )

        print(f"Case {number}: {name}")

        print(
            f"\n  Previous mode: "
            f"{result['previous_mode']}"
        )

        print(f"\n  Temperature: {temperature} C")

        print(f"\n  Occupied: {occupied}")

        print(
            f"\n  Target mode: "
            f"{result['target_mode']}"
        )

        print(f"\n  Outcome: {command_result}")

        print("\n" + "-" * 70 + "\n")

    print(f"Total cases: {len(test_cases)}")

    print(f"\nCommands sent: {commands_sent}")

    print(
        f"\nCommands avoided: "
        f"{len(test_cases) - commands_sent}"
    )


# ==================================================
# MAIN PROGRAM
# ==================================================

def main():

    display_peas()

    display_environment_classifications()

    run_room_controller()


if __name__ == "__main__":

    main()