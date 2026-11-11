import json
from pathlib import Path
import sys
import unittest

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Human-study fixtures require NumPy") from exc

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from brain.minds_analysis import natural_basis,summarize
from brain.minds_remote import midranks,resolve_country,map_coordinates
from brain import minds_remote


class MindsTests(unittest.TestCase):
    def test_midrank_ties_reference(self):
        np.testing.assert_allclose(midranks(np.array([2,1,2,4])),[.5,.125,.5,.875])
        np.testing.assert_allclose(midranks(np.array([1,1,1])),[.5,.5,.5])

    def test_country_weighting(self):
        # Unequal population sizes do not enter the country estimator.
        self.assertAlmostEqual(summarize([.2,.8])['mean'],.5)
        self.assertAlmostEqual(summarize([.2,.8])['mean'],np.mean([.2,.8]))

    def test_natural_boundary_linearity(self):
        basis=natural_basis([-10,-9,-8,90,91,92])
        np.testing.assert_allclose(basis[0]-2*basis[1]+basis[2],0,atol=1e-12)
        np.testing.assert_allclose(basis[3]-2*basis[4]+basis[5],0,atol=1e-12)
        self.assertEqual(natural_basis(range(18,65)).shape,(47,6))

    def test_conflicting_geography(self):
        self.assertEqual(resolve_country('Canada','United States of America'),('','conflict'))
        self.assertEqual(resolve_country('Canada','Canada'),('Canada','agreement'))
        self.assertEqual(resolve_country('','Canada'),('Canada','coordinates_only'))
        self.assertEqual(resolve_country('',''),('','unresolved'))

    def test_coordinate_order_holes_territories(self):
        previous=getattr(minds_remote,'GEO',None)
        minds_remote.GEO={'geography':[{'name':'Territory','rings':[[[20,10],[22,10],[22,12],[20,12],[20,10]],[[20.8,10.8],[21.2,10.8],[21.2,11.2],[20.8,11.2],[20.8,10.8]]]}]}
        try:
            result=map_coordinates(['[10.5,20.5]','[20.5,10.5]','[11,21]','[95,20]','bad'])
            self.assertEqual(result['[10.5,20.5]'][0],'Territory')
            self.assertEqual(result['[20.5,10.5]'][0],'')
            self.assertEqual(result['[11,21]'][0],'')
            self.assertFalse(result['[95,20]'][1]);self.assertFalse(result['bad'][1])
        finally:minds_remote.GEO=previous

    def test_ambiguous_aliases(self):
        d=json.loads((ROOT/'evidence/brain/geographic-crosswalk.json').read_text())
        self.assertNotIn('georgia',d['aliases'])
        self.assertEqual(d['country_aliases']['georgia'],'Georgia')
        self.assertEqual(d['country_aliases']['guadeloupe'],'Guadeloupe')
        self.assertEqual(d['aliases']['ontario'],'Canada')

class GssDesignTests(unittest.TestCase):
    def test_decomposition_matches_direct_covariance(self):
        from brain.minds_gss import sufficient,decomposition_from_sums
        x=np.array([1.,2.,4.,6.]);c=np.array([0.,1.,2.,3.]);w=np.array([1.,2.,3.,4.])
        moments=sufficient(x,c,w,np.zeros(4,int),1)
        result=decomposition_from_sums(moments)
        mx=np.average(x,weights=w);sd=np.sqrt(np.average((x-mx)**2,weights=w));z=(x-mx)/sd
        total=np.average(z*c,weights=w)/np.average(c,weights=w)
        parent=c>0;access=np.average(z[parent],weights=w[parent])
        np.testing.assert_allclose(result,[total,access,total-access],atol=1e-12)

    def test_full_design_support_and_singleton_exclusion(self):
        from brain.minds_gss import design_multiplier
        d={'year':np.array([2000,2000,2000]),'vstrat':np.array([1,1,2]),'vpsu':np.array([1,2,1])}
        valid,groups,replicates,info=design_multiplier(d,np.ones(3))
        np.testing.assert_equal(valid,[True,True,False]);self.assertEqual(info['singleton_strata_excluded'],1)
        np.testing.assert_allclose(replicates.sum(axis=1),2)
        self.assertTrue(np.all(replicates[:,2]==0))


class DemographicBaselineTests(unittest.TestCase):
    def test_metric_rank_ties_constant_and_negative_skill(self):
        from brain.minds_demographic_remote import metrics,midrank
        np.testing.assert_allclose(midrank(np.array([1,1,3])),[1/3,1/3,5/6])
        result=metrics(np.array([1.,2.,3.]),np.array([1.,1.,1.]))
        self.assertIsNone(result['pearson_r']);self.assertIsNone(result['spearman_r'])
        self.assertAlmostEqual(result['r2'],-1.5)

    def test_fit_never_uses_validation_or_test_targets(self):
        from brain.minds_demographic_remote import fit_baselines
        profiles=[]
        for i in range(10000):
            age=18+(7*i)%82;gender='Male' if i%2 else 'Female';split='train' if i<8000 else 'validation' if i<9000 else 'test'
            profiles.append({'age':age,'gender':gender,'split':split,'beauty_score':4+.01*age+.5*(gender=='Male')})
        original,parameters=fit_baselines(profiles)
        changed=[dict(row,beauty_score=row['beauty_score']+(3 if row['split']=='test' else 2 if row['split']=='validation' else 0)) for row in profiles]
        modified,second=fit_baselines(changed)
        self.assertEqual(parameters,second)
        self.assertEqual(original['age_gender_spline']['train_metrics'],modified['age_gender_spline']['train_metrics'])
        self.assertLess(original['age_gender_spline']['test_metrics']['beauty_mae'],1e-10)
        self.assertGreater(modified['age_gender_spline']['test_metrics']['beauty_mae'],2.99)


if __name__=='__main__':unittest.main()
