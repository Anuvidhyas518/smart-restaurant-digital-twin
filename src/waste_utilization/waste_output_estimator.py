def estimate_utilization(
    waste_kg,
    organic_fraction=1.0,
    volatile_solids_fraction=0.20,
    methane_fraction=0.60,
    compost_recovery_rate=0.50
):
    """
    Estimate potential outputs from food waste.

    Parameters
    ----------
    waste_kg : float
        Total available food waste in kg.

    organic_fraction : float
        Fraction of total waste that is suitable organic
        feedstock for biological processing.

    volatile_solids_fraction : float
        Fraction of organic feedstock represented as
        volatile solids (VS).

    methane_fraction : float
        Assumed methane fraction of produced biogas.

    compost_recovery_rate : float
        Scenario assumption for the fraction of compostable
        input remaining as finished compost.

    Returns
    -------
    dict
        Estimated quantities and ranges.
    """

    # ------------------------------------------
    # Basic validation
    # ------------------------------------------

    if waste_kg < 0:
        raise ValueError(
            "Waste quantity cannot be negative."
        )

    if not 0 <= organic_fraction <= 1:
        raise ValueError(
            "Organic fraction must be between 0 and 1."
        )

    if not 0 <= volatile_solids_fraction <= 1:
        raise ValueError(
            "Volatile-solids fraction must be between 0 and 1."
        )

    if not 0 < methane_fraction <= 1:
        raise ValueError(
            "Methane fraction must be greater than 0 and at most 1."
        )

    if not 0 <= compost_recovery_rate <= 1:
        raise ValueError(
            "Compost recovery rate must be between 0 and 1."
        )

    # ------------------------------------------
    # Organic feedstock
    # ------------------------------------------

    organic_waste_kg = (
        waste_kg * organic_fraction
    )

    # ------------------------------------------
    # Volatile solids
    # ------------------------------------------

    volatile_solids_kg = (
        organic_waste_kg *
        volatile_solids_fraction
    )

    # ------------------------------------------
    # Methane yield range
    #
    # Literature range:
    # 0.27 - 0.642 m3 CH4 / kg VS
    # ------------------------------------------

    methane_low_m3 = (
        volatile_solids_kg * 0.27
    )

    methane_high_m3 = (
        volatile_solids_kg * 0.642
    )

    # ------------------------------------------
    # Convert methane potential to total
    # biogas potential using user assumption
    # for methane fraction.
    #
    # This is an estimate, not a measured value.
    # ------------------------------------------

    biogas_low_m3 = (
        methane_low_m3 /
        methane_fraction
    )

    biogas_high_m3 = (
        methane_high_m3 /
        methane_fraction
    )

    # ------------------------------------------
    # Compost scenario
    # ------------------------------------------

    compost_potential_kg = (
        organic_waste_kg *
        compost_recovery_rate
    )

    return {
        "organic_waste_kg": organic_waste_kg,
        "volatile_solids_kg": volatile_solids_kg,
        "methane_low_m3": methane_low_m3,
        "methane_high_m3": methane_high_m3,
        "biogas_low_m3": biogas_low_m3,
        "biogas_high_m3": biogas_high_m3,
        "compost_potential_kg": compost_potential_kg
    } 