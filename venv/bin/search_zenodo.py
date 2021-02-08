#!/home/wolfram/projekte/enzyme_cost_minimization/github/marco-corrao/metabolic_modelling/venv/bin/python
""" examples of searching Zenodo"""
import pyzenodo3


def main():
    zen = pyzenodo3.Zenodo()

    Recs = zen.search("scivision")
    assert Recs is not None
    # %%
    Rec = zen.find_record_by_github_repo("scivision/lowtran")

    vers = Rec.get_versions_from_webpage()

    print(vers)


if __name__ == "__main__":
    main()
