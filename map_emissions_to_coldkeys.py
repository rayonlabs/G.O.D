#!/usr/bin/env python3
import json
import asyncio
from validator.db.database import get_db
from validator.db.sql.nodes import get_all_nodes_for_netuid

# Your emissions data
emissions_data = {
    "netuid": 56,
    "total_emissions_owed": 3542.3999999999987,
    "miners_affected": 60,
    "epochs_processed": 24,
    "emission_per_epoch": 147.6,
    "expected_total": 3542.3999999999996,
    "difference": 9.094947017729282e-13,
    "emissions_by_miner": {
        "5GpxMEQYVnsLZE93FYh76w5MeFjufBprYfPuf8xuUDGoWp82": 53.338964519759735,
        "5GYjsCTLxCC9ELVP6h3YgtgnHmakgbKNp7X5CNRwzThfpSEm": 107.2021038068916,
        "5Fc44E5mqrAyRf58hUuyeVXwLHR3tFQkeg5mh12QEGKGN5PA": 119.25674213245392,
        "5CA2H88YhkyW5urHRn8RxdDUK6h4eEa9w897CuqbRgsttwfV": 1.408084529225661,
        "5Gph1wxbdwzscQp4iZZEAMCRPDRSzzAp814LsD5sLVw3mFCG": 26.688600688897864,
        "5HNVS6zjj84CN1ehRrwjYduVG3JWwWnPGsLhAmLoY1iX1fNF": 177.37902531753477,
        "5GWVcNj2iTM84VqAB5hK2knF48NjQT9WCLriJ9pJMJkuPUkq": 7.2621548381631,
        "5FhCWMDSZJMoNxD1QryrCrNAJngfhrzDECmWnYco4JWR34LB": 86.11862435573121,
        "5CUY3uo3eAdQqq8gbNdgA6upbo4v7DRA9CkuAJmQAyVUY4zT": 8.172135506403272,
        "5CXVZtYp9BHhbjUQ4EF1JYrtRNpvbfHKHXciGUC1qPMKRK1R": 14.857157162817604,
        "5Cr1E2Ds5Dc5VGi7oSWatRkQycshGbGGtZNduaYwhEBdvnaX": 14.268196725025234,
        "5ERXnkFwgD3mb9BiUyFwYmW1vb1WoRNKdbKV9tCsJnCGjmDq": 5.457133418383244,
        "5DLkfAsZfex9npNhKVCWR6qGYRmvZVa6bvVLkYdf2nkSqBPu": 256.0711322156117,
        "5FxZzeoVp46Pn5XkXCguAq6MH1gQwkasbgUCPUAMbZ4zG54X": 6.797588452362124,
        "5HBHUtS9uSr4v5kzwGsHF25WjPdfuGaP9vccMhHPQobXd9TZ": 10.901593473787399,
        "5GxCXUgVr5Mgvk6hPWcaUYc5yejwESBumpz69WgsQQhbaoc7": 306.511515969087,
        "5HCE2wvuKyXbjVvTWwBFW2QhLuyombykiQ5rkPaRUonLCRa7": 26.055080387400825,
        "5CfUfLSvpcs3x4NXj4hmQznB5F7VqbP1nimNGEDWeHTp6dp3": 9.061240029997986,
        "5E299qqve4qePwLJYk5AiXTxxmvvrTDjpqGCL9WwEQD4v8S6": 4.262597562111554,
        "5GKHkHdgDDSWtfCepeqQsPTdma8Veht3SijGGXM4qms71NWT": 3.467964162519957,
        "5CcY6aQqz34Y1QnoZvQouFDW4iSnBXfAPBA95rnHoMGeiXoz": 14.706896657851273,
        "5HagoXpam6KscfXbZP6FxmThuYkHwXXCmVeFj8PjaD8fqeWz": 12.051573515286627,
        "5DLQekjgFeHFFqh3DjNUbPwCt3pAnnVGTXRtxrLz97RJjrEC": 193.50576443473972,
        "5F6gzidLxREkhaRTBXDYEVz7fybAurxKrkj3YnBD9ySdsSrg": 81.59242865832863,
        "5GCG2QRtmNUrYqh8djZzD8QTPxno8jHwQC2Je4P4Aj8GKHaq": 103.88806920579752,
        "5FpPHuqZenoN2gxZ9r8w6VmCjdG68ctaNGg61bCFz57oi2oJ": 159.01006204928174,
        "5HWdtehpMTHNpRKxkN2bDj72K2YPimPw9Denbx9nrkpfajoR": 14.685807574108573,
        "5GH7pbnYTNevMBtCiUz4Gin7mmsGnqNatVUHZYqfbbtsbnkG": 35.235727518055484,
        "5GEaWTcYnamnuoGznRm3jftCpu2FumUYon2AiNn8TbRBDfit": 168.51021130736126,
        "5DaCpJPyCTS6mYYn9m9aKsQXy6Whxi7hnE9uzneARuvG3jRF": 37.312096665622256,
        "5CXShfZG2W7pfKcASKSuYnPUQqLykyE53PNDjpT8enopT1r5": 62.07941514611249,
        "5DCx43cgGoPUW8p8zMB7JzQzfsFRi2PRULdVYkdySjb88wRM": 32.79075719491375,
        "5D22pEL7SUQXakCseL3Vgj2xfBkMPXxtKu4B7ETdXfjNcu9A": 140.87071640827017,
        "5DqAVk38YCFyfN3iAxcUYbK9hQQtYFqiVfgbrNesYvjNgc3K": 14.128188835899989,
        "5GeV1LeSCU7EqGgpPPu3mGJd7M91eYGGDHgViv8QT7dPxdAE": 10.321637071993019,
        "5HHHfr2R2rnauhYQGkLnnAq7MZorGp6AP9vpLXWNN4MvMR9X": 2.9033193275226474,
        "5E7poKRYVRy7r21Fo79xXp7gTBzbyuEB82gpcPWz9d6LWDy8": 8.07771282151922,
        "5CANkqdD5QsQd8Nu5p9ANjYLG8rU2NKyq3a1LLK4XgzgMjiP": 17.839425868786794,
        "5D2EUQvKyzbrK54CDY1UCs4TmdW7ZcKWKTpuFFxxZXKmm5jz": 20.015917957520294,
        "5Da1vaJZ3EHGNRhD4dcbaqDVwqtX4ckPYugHjHwgEEjwxera": 2.1194322588563095,
        "5Dm2FaUFWs4UZQasbX4aA7D7MGXxRvGYQETnpJWfWieXgxQ4": 58.813866068474134,
        "5DS8ste6pSBv2EsNX7D6K7GEmHZ6Ru5MtTiDXGzPn7kFWbag": 10.597660392859542,
        "5GgRyjncAxjvL9s3XGoUvD9ftpFm9Y3fVWbsnFs151RGV5zF": 1.880131768401164,
        "5FW8moDY5XERUatSKkU5V1eiC3dPzEBv16YxYcP7cr9geJVK": 24.93973905209012,
        "5D73FgRPZHJurUWdMrWmhMRzweXcgMiwujwhjkvNPLH9A5Fq": 3.3136378070480816,
        "5DGBHANau8Hm6XGYyuSAd3X4rG5KJsxGs75kJPNoFje4AzRa": 345.07061123093575,
        "5G1nGrP2Pq3H8DyLt4kfemBuvrDhg3DNJmrwj3ayHZSDihyL": 68.19835722548596,
        "5FLKAQpUDDERFBVLf48zjvPg6uE6qPNLnp7iGW8VEthmPace": 17.120660707932934,
        "5DsnVdHur2M9dnYax2A5DPhQ6XkvdzmcyrgkMbQyAG7skGdk": 56.87301990787107,
        "5HL9KhhFW6YRgLFFnpqk4JwCUQtZfoeWjWMdPzuCuzQqy5oi": 120.0643710794127,
        "5DvX12NPMaqCAz4c8R1gPEDZDdqQceiGUHoED8eQ87XpyKhN": 29.152782420804513,
        "5F4nqBQLNpYy4H3vZQAdRin1hEtd3H6adcefndw3k7fzGHaQ": 222.06704399858626,
        "5GKMveAXkfcW73dEDBE18FdZ2XjkwcURhctWHvz8KUGpJkgz": 128.84648691144562,
        "5H71QKP27EErKAMueM2jbh9LEofbrS6qQPZ5t65FZQbEtnyx": 8.885338091414203,
        "5E9Lpf5Dz1kZxpSDYqKqYxvhGwQBZy4xpatFSs2VdL8GUiPq": 8.37712203494046,
        "5E9M2U3QRUwwEyTAPjPoZS951mUGL2MsVNDNJuovCf6Z3RJi": 16.16367566957657,
        "5EJ4XUwxromJik5x5K8Rx1veMBbc9BpouqZZfJoXugUa9r8y": 8.647005355636148,
        "5DyfYyy58FGB5qZrkKkYiqj8vijFi6kHtmrbJNNdtKCryETY": 2.9016429197983333,
        "5CmF4oHhJ3pWUuUrjM3DChTi6J7ZCwQjob4BF5jftLyLbiej": 9.023147290458239,
        "5G3d5vNCQkf9Z6LthgZrTP4C72RYDmn75CYduGGC2m2CXHvi": 25.280904334834094
    }
}

async def main():
    async with get_db() as db:
        # Get all nodes for netuid 56
        nodes = await get_all_nodes_for_netuid(db, 56)
        
        # Create hotkey to coldkey mapping
        hotkey_to_coldkey = {}
        for node in nodes:
            hotkey_to_coldkey[node['hotkey']] = node['coldkey']
        
        # Map emissions to coldkeys
        coldkey_emissions = {}
        unmapped_hotkeys = []
        
        for hotkey, emission in emissions_data['emissions_by_miner'].items():
            if hotkey in hotkey_to_coldkey:
                coldkey = hotkey_to_coldkey[hotkey]
                if coldkey not in coldkey_emissions:
                    coldkey_emissions[coldkey] = 0
                coldkey_emissions[coldkey] += emission
            else:
                unmapped_hotkeys.append(hotkey)
        
        # Sort by emissions (descending)
        sorted_coldkeys = sorted(coldkey_emissions.items(), key=lambda x: x[1], reverse=True)
        
        # Print results
        print(f"Emissions by Coldkey (netuid {emissions_data['netuid']})")
        print("=" * 80)
        print(f"{'Coldkey':<55} {'Emissions':>20}")
        print("-" * 80)
        
        total_mapped = 0
        for coldkey, emission in sorted_coldkeys:
            print(f"{coldkey:<55} {emission:>20.6f}")
            total_mapped += emission
        
        print("-" * 80)
        print(f"{'Total Mapped:':<55} {total_mapped:>20.6f}")
        print(f"{'Total Emissions:':<55} {emissions_data['total_emissions_owed']:>20.6f}")
        print(f"{'Number of Coldkeys:':<55} {len(coldkey_emissions):>20}")
        
        if unmapped_hotkeys:
            print("\nUnmapped Hotkeys:")
            for hotkey in unmapped_hotkeys:
                print(f"  {hotkey}: {emissions_data['emissions_by_miner'][hotkey]:.6f}")
        
        # Save to JSON file
        output = {
            "netuid": emissions_data["netuid"],
            "total_emissions": emissions_data["total_emissions_owed"],
            "coldkey_emissions": dict(sorted_coldkeys),
            "unmapped_hotkeys": {hk: emissions_data['emissions_by_miner'][hk] for hk in unmapped_hotkeys}
        }
        
        with open("emissions_by_coldkey.json", "w") as f:
            json.dump(output, f, indent=2)
        
        print("\nResults saved to emissions_by_coldkey.json")

if __name__ == "__main__":
    asyncio.run(main())