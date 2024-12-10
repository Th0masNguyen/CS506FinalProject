from flask import Flask, render_template, request
from model import gbCalendarEvents, cbCalendarEvents, gbPredictedSaleDate, cbPredictedSaleDate

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def toggle_content():
    # Default settings
    selected_images = ["groundBeefWeeksUntilHisto.png", "groundBeefIntArrival.png"]
    selected_calendar_events = gbCalendarEvents
    predicted_sale_date = gbPredictedSaleDate
    selected_option_name = "Ground Beef"

    if request.method == "POST":
        selected_option = request.form.get("option")
        if selected_option == "Ground Beef":
            selected_images = ["groundBeefWeeksUntilHisto.png", "groundBeefIntArrival.png"]
            selected_calendar_events = gbCalendarEvents
            predicted_sale_date = gbPredictedSaleDate
            selected_option_name = "Ground Beef"
        elif selected_option == "Chicken Breasts":
            selected_images = ["chickenBreastsWeeksUntilHisto.png", "chickenBreastsIntArrival.png"]
            selected_calendar_events = cbCalendarEvents
            predicted_sale_date = cbPredictedSaleDate
            selected_option_name = "Chicken Breasts"

    return render_template(
        "toggle.html",
        images=selected_images,
        events=selected_calendar_events,
        option_name=selected_option_name,
        sale_date=predicted_sale_date
    )

if __name__ == "__main__":
    app.run(debug=True)
