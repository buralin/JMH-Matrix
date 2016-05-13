package org.wallerlab.yoink.domain;

import java.util.Random;

public class SettingUpData {
	public static float[] createRandomFloatData(int n)
    {
        Random random = new Random();
        float x[] = new float[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = random.nextFloat();
        }
        return x;
    }

}
