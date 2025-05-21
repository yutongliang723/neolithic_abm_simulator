import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))
from app.vec import Vec1
from app.village import Village
from app.utils import generate_random_village

class TestVillage(unittest.TestCase):
    def test_generate_random_village(self):
        params = {
            "num_house": 10,
            "land_cells": 50,
            "year": 10,
            "food_expiration_steps": 5,
            "land_recovery_rate": 0.05,
            "land_max_capacity": 10,
            "initial_quality": 5,
            "fallow_period": 2,
            "luxury_goods_in_village": 0,
            "fertility_scaler": 2,
            "demog_file": "demog_vectors_scaled.csv"
        }
        vec1_instance = Vec1(params)

        village = generate_random_village(
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

        self.assertEqual(len(village.households), 10)

    def test_simulation_runs_per_year(self):
        params = {
            "num_house": 5,
            "land_cells": 20,
            "year": 5,
            "prod_multiplier": 2,
            "fishing_discount": 2,
            "spare_food_enabled": True,
            "fallow_farming": False,
            "emigrate_enabled": True,
            "trading_enabled": True,
            "land_recovery_rate": 0.1,
            "food_expiration_steps": 3,
            "land_depreciate_factor": 0.05,
            "max_member": 6,
            "fallow_period": 2,
            "marriage_from": 14,
            "marriage_to": 60,
            "bride_price_ratio": 0.4,
            "bride_price": 1,
            "land_max_capacity": 10,
            "initial_quality": 5,
            "exchange_rate": 3,
            "luxury_good_storage": 0,
            "storage_ratio_low": 0.2,
            "storage_ratio_high": 1.5,
            "land_capacity_low": 1,
            "excess_food_ratio": 1.5,
            "trade_back_start": 20,
            "lux_per_year": 10,
            "fertility_scaler": 2,
            "work_scale": 5,
            "luxury_goods_in_village": 0,
            "demog_file": "demog_vectors_scaled.csv",
            "prob_emigrate": 0.1,
            "farming_counter_max": 5,
            "conditions": {
                "use_fertility": True,
                "check_gender": True,
                "check_marital_status": True,
                "check_land": False,
                "exceed_member": True
            }
        }

        vec1_instance = Vec1(params)

        village = generate_random_village(
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

        for _ in range(params["year"]):
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

        self.assertTrue(len(village.households) > 0)

if __name__ == '__main__':
    unittest.main()