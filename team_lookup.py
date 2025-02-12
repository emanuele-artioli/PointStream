import os
import re
import argparse
import requests

def fetch_team_data(team_name: str, year: int):
    """
    Fetch real data from Wikipedia for the specified team and year.
    This is a naive example and may need adaptation for different team pages.

    Returns a dictionary containing jerseys, players, numbers, etc.
    """
    base_url = "https://en.wikipedia.org/w/api.php"

    # Step 1: Search for the team's page
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": f"{team_name} {year} football team",
        "format": "json"
    }
    search_response = requests.get(base_url, params=search_params).json()
    search_results = search_response["query"]["search"]
    if not search_results:
        return {
            "team": team_name,
            "year": year,
            "jerseys": {},
            "players": []
        }

    # Attempt to retrieve the first matching page
    page_title = search_results[0]["title"]

    # Step 2: Get the page content in Wikitext
    page_params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "titles": page_title,
        "format": "json"
    }
    page_response = requests.get(base_url, params=page_params).json()
    pages = page_response["query"]["pages"]
    page_content = ""
    for _, val in pages.items():
        if "revisions" in val:
            page_content = val["revisions"][0]["slots"]["main"]["*"]
            break

    # Step 3: Find or approximate a "Current squad" or similar section
    # This is a highly naive approach relying on certain wikitext patterns.
    # Real usage would require more robust parsing or a specialized library.
    lines = page_content.split("\n")
    squad_lines = []
    in_squad_section = False
    for line in lines:
        if re.search(r"(Current squad|Squad)", line, re.IGNORECASE):
            in_squad_section = True
            continue
        if in_squad_section:
            # Ends if we encounter another heading
            if line.startswith("=="):
                break
            squad_lines.append(line)

    # A naive approach to glean jersey number and names from bullet or table lines
    player_pattern = re.compile(r"\|\s*(\d+)\s*\|\s*(.*?)\|")  # e.g. "| 10 | Lionel Messi |"

    # Simple jersey dictionary for demonstration (we can't reliably parse these from text)
    # In real practice, you might look for kit templates or images.
    jerseys = {
        "home": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/67/Soccer_current_event_template.svg/120px-Soccer_current_event_template.svg.png",
        "away": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Football_kit_template.svg/100px-Football_kit_template.svg.png",
        "goalie": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/Goalkeeper_kit.svg/90px-Goalkeeper_kit.svg.png"
    }

    players = []
    for squad_line in squad_lines:
        match = player_pattern.search(squad_line)
        if match:
            number = match.group(1)
            player_name = match.group(2).strip()
            # Remove wikilink syntax, e.g. [[Lionel Messi|Messi]]
            player_name = re.sub(r"\[\[|\]\]", "", player_name)
            player_name = re.sub(r"\|.*", "", player_name)  # remove anything after '|'
            players.append({
                "name": player_name.strip(),
                "number": number.strip(),
                "face_image": ""  # Real face images require more advanced logic
            })

    return {
        "team": team_name,
        "year": year,
        "jerseys": jerseys,
        "players": players
    }

def main():
    parser = argparse.ArgumentParser(description="Fetch soccer team data from Wikipedia.")
    parser.add_argument("--team_name", type=str, required=True, help="Name of the soccer team.")
    parser.add_argument("--year", type=int, required=True, help="Year of interest.")
    args = parser.parse_args()

    data = fetch_team_data(args.team_name, args.year)
    # Display or save data as needed
    print(f"Team: {data['team']} ({data['year']})")
    print("Jerseys:")
    for jersey_type, jersey_url in data["jerseys"].items():
        print(f"  {jersey_type.title()} Jersey: {jersey_url}")

    print("Players:")
    for player in data["players"]:
        print(f"  Name: {player['name']}, Number: {player['number']}, Face: {player['face_image']}")

if __name__ == "__main__":
    main()