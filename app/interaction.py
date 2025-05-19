from flask import Flask, request, jsonify, render_template, send_from_directory
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import threading
import math
import os
import json
import time
import shutil
from demog_scale import demog_scale
from vec import Vec1
from village import Village
import utils

app = Flask(__name__)
progress_percent = 0
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_results', 'website')
LATEST_FOLDER = os.path.join(RESULTS_FOLDER, 'latest')
os.makedirs(LATEST_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run_results/website/<path:filename>')
def serve_results(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

@app.route('/progress')
def get_progress():
    global progress_percent
    return jsonify({"percent": progress_percent})

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    global progress_percent
    progress_percent = 0

    data = request.json
    required_params = [
        "num_house", "year", "land_cells",
        "spare_food_enabled", "fallow_farming", "emigrate_enabled",
        "land_recovery_rate", "food_expiration_steps", "trading_enabled"
    ]

    for param in required_params:
        if param not in data:
            return jsonify({"error": f"Missing required parameter: {param}"}), 400

    try:
        params = {
            "num_house": int(data["num_house"]),
            "year": int(data["year"]),
            "land_cells": int(data["land_cells"]),
            "prod_multiplier": 2,
            "fishing_discount": 2,
            "spare_food_enabled": data["spare_food_enabled"] == "true",
            "fallow_farming": data["fallow_farming"] == "true",
            "emigrate_enabled": data["emigrate_enabled"] == "true",
            "trading_enabled": data["trading_enabled"] == "true",
            "land_recovery_rate": float(data["land_recovery_rate"]),
            "food_expiration_steps": int(data["food_expiration_steps"]),
            "land_depreciate_factor": float(data["land_depreciate_factor"]),
            "max_member": int(data["max_member"]),
            "fallow_period": 2,
            "marriage_from": 14,
            "marriage_to": 60,
            "bride_price_ratio": 0.4,
            "bride_price": 1,
            "land_max_capacity": 20,
            "initial_quality": 5,
            "exchange_rate": 3,
            "luxury_good_storage": 0,
            "storage_ratio_low": 0.2,
            "storage_ratio_high": 1.5,
            "land_capacity_low": 1,
            "excess_food_ratio": 1.5,
            "trade_back_start": 20,
            "lux_per_year": 15,
            "fertility_scaler": 2,
            "work_scale": 5,
            "luxury_goods_in_village": 0,
            "demog_file": "demog_vectors_scaled.csv",
            "prob_emigrate": 0.2,
            "farming_counter_max": 10,
            "conditions": {
                "use_fertility": True,
                "check_gender": True,
                "check_marital_status": True,
                "check_land": False,
                "exceed_member": True
            }
        }
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid parameter: {e}"}), 400

    thread = threading.Thread(target=run_simulation_task, args=(params,))
    thread.start()
    return jsonify({"status": "started", "folder": "latest"})

def run_simulation_task(params):
    global progress_percent

    # Run setup
    demog_scale()
    vec1_instance = Vec1(params)

    village = utils.generate_random_village(
        num_households=params["num_house"],
        num_land_cells=params["land_cells"],
        vec1_instance=vec1_instance,
        food_expiration_steps=params["food_expiration_steps"],
        land_recovery_rate=params["land_recovery_rate"],
        land_max_capacity=params["land_max_capacity"],
        initial_quality=params["initial_quality"],
        fallow_period=params["fallow_period"],
        luxury_goods_in_village=params["luxury_goods_in_village"]
    )

    village.initialize_network()
    village.initialize_network_relationship()

    for i in range(params["year"]):
        village.run_simulation_step(
            vec1_instance=vec1_instance,
            prod_multiplier=params["prod_multiplier"],
            fishing_discount=params["fishing_discount"],
            fallow_period=params["fallow_period"],
            food_expiration_steps=params["food_expiration_steps"],
            marriage_from=params["marriage_from"],
            marriage_to=params["marriage_to"],
            bride_price_ratio=params["bride_price_ratio"],
            bride_price=params["bride_price"],
            exchange_rate=params["exchange_rate"],
            storage_ratio_low=params["storage_ratio_low"],
            storage_ratio_high=params["storage_ratio_high"],
            land_capacity_low=params["land_capacity_low"],
            max_member=params["max_member"],
            excess_food_ratio=params["excess_food_ratio"],
            trade_back_start=params["trade_back_start"],
            lux_per_year=params["lux_per_year"],
            land_depreciate_factor=params["land_depreciate_factor"],
            fertility_scaler=params["fertility_scaler"],
            work_scale=params["work_scale"],
            conditions=params["conditions"],
            prob_emigrate=params["prob_emigrate"],
            emigrate_enabled=params["emigrate_enabled"],
            spare_food_enabled=params["spare_food_enabled"],
            fallow_farming=params["fallow_farming"],
            trading_enabled=params["trading_enabled"],
            farming_counter_max=params["farming_counter_max"]
        )
        progress_percent = int((i + 1) / params["year"] * 100)

    # Clean latest/
    if os.path.exists(LATEST_FOLDER):
        shutil.rmtree(LATEST_FOLDER)
    os.makedirs(LATEST_FOLDER, exist_ok=True)

    # Save parameters and output files
    with open(os.path.join(LATEST_FOLDER, "parameters.json"), "w") as f:
        json.dump(params, f, indent=4)

    # Prepare paths
    results_svg = os.path.join(LATEST_FOLDER, "results.svg")
    results_second_svg = os.path.join(LATEST_FOLDER, "results_second.svg")
    results_csv = os.path.join(LATEST_FOLDER, "simulation_results.csv")
    animation_gif = os.path.join(LATEST_FOLDER, "simulation.gif")

    # Generate plots and animation
    village.plot_simulation_results(results_svg, results_csv, vec1_instance)
    village.plot_simulation_results_second(results_second_svg)
    village.generate_animation(animation_gif, grid_dim=math.ceil(math.sqrt(params['land_cells'])))

    # ✅ Wait until all four files are physically on disk
    expected_files = [results_svg, results_second_svg, results_csv, animation_gif]
    timeout = 20  # seconds max to wait
    elapsed = 0
    poll_interval = 0.5

    while elapsed < timeout:
        if all(os.path.isfile(f) and os.path.getsize(f) > 0 for f in expected_files):
            break
        time.sleep(poll_interval)
        elapsed += poll_interval

    # Only now, report simulation as complete
    progress_percent = 100

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

import os
port = int(os.environ.get("PORT", 5000))
app.run(host='0.0.0.0', port=port)