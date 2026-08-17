def calculate_period_impact(
    waste_kg,
    organic_fraction=1.0,
    volatile_solids_fraction=0.20,
    methane_fraction=0.60,
    energy_per_m3_biogas_kwh=6.0,
    energy_value_per_kwh=8.0,
    disposal_cost_per_kg=0.0
):
    if waste_kg < 0:
        raise ValueError("Waste quantity cannot be negative.")

    if not 0 <= organic_fraction <= 1:
        raise ValueError("Organic fraction must be between 0 and 1.")

    if not 0 <= volatile_solids_fraction <= 1:
        raise ValueError(
            "Volatile solids fraction must be between 0 and 1."
        )

    if not 0 < methane_fraction <= 1:
        raise ValueError(
            "Methane fraction must be greater than 0 and at most 1."
        )

    if not 0 <= disposal_cost_per_kg:
        raise ValueError(
            "Disposal cost cannot be negative."
        )

    organic_waste = waste_kg * organic_fraction

    volatile_solids = (
        organic_waste * volatile_solids_fraction
    )

    methane_low = volatile_solids * 0.27
    methane_high = volatile_solids * 0.642

    biogas_low = methane_low / methane_fraction
    biogas_high = methane_high / methane_fraction

    energy_low = (
        biogas_low * energy_per_m3_biogas_kwh
    )

    energy_high = (
        biogas_high * energy_per_m3_biogas_kwh
    )

    energy_value_low = (
        energy_low * energy_value_per_kwh
    )

    energy_value_high = (
        energy_high * energy_value_per_kwh
    )

    disposal_saving = (
        waste_kg * disposal_cost_per_kg
    )

    total_value_low = (
        energy_value_low + disposal_saving
    )

    total_value_high = (
        energy_value_high + disposal_saving
    )

    return {
        "waste_kg": waste_kg,
        "organic_waste_kg": organic_waste,
        "volatile_solids_kg": volatile_solids,
        "methane_low_m3": methane_low,
        "methane_high_m3": methane_high,
        "biogas_low_m3": biogas_low,
        "biogas_high_m3": biogas_high,
        "energy_low_kwh": energy_low,
        "energy_high_kwh": energy_high,
        "energy_value_low": energy_value_low,
        "energy_value_high": energy_value_high,
        "disposal_saving": disposal_saving,
        "total_value_low": total_value_low,
        "total_value_high": total_value_high
    } 