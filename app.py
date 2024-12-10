from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def toggle_content():
    # Default settings
    selected_images = ["groundBeefWeeksUntilHisto.png", "groundBeefIntArrival.png"]
    selected_calendar_events = [
        {"title": "Event 1 - Option 1", "start": "2024-12-11", "end": "2024-12-11"},
        {"title": "Event 2 - Option 1", "start": "2024-12-15"},
    ]

    if request.method == "POST":
        selected_option = request.form.get("option")
        if selected_option == "option1":
            selected_images = ["groundBeefWeeksUntilHisto.png", "groundBeefIntArrival.png"]
            selected_calendar_events = [
                {"title": "Event 1 - Option 1", "start": "2024-12-11", "end": "2024-12-11"},
                {"title": "Event 2 - Option 1", "start": "2024-12-15"},
            ]
        elif selected_option == "option2":
            selected_images = ["chickenBreastsWeeksUntilHisto.png", "chickenBreastsIntArrival.png"]
            selected_calendar_events = [
                {"title": "Event 1 - Option 2", "start": "2024-12-20", "end": "2024-12-20"},
                {"title": "Event 2 - Option 2", "start": "2024-12-25"},
            ]

    return render_template(
        "toggle.html",
        images=selected_images,
        events=selected_calendar_events
    )

if __name__ == "__main__":
    app.run(debug=True)
