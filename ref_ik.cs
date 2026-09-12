using System;
using UnityEngine;

public class ArmController : MonoBehaviour
{
    [SerializeField] private Transform target; // 追従する物体
    [SerializeField] private Transform endEffector; // 手先の物体
    [SerializeField] private Transform root; // ロボットアームの根元
    [SerializeField] private Transform[] joints; // ロボットアームの各関節（6つ）
    [SerializeField] private Transform rightFinger;
    [SerializeField] private Transform leftFinger;
    [SerializeField] private float ikRange = 0.5f; // 手先の位置がこの範囲内に収まるとIKを計算する
    [SerializeField] private bool enableIK = true; // IK計算の有効/無効
    [SerializeField] private bool isLeftHand = false; // 左手かどうか
    private double[] theta = new double[6];
    private float triggerPower;
    private float L1, L2, L3, L4, L5, L6;
    private float fingerPower = 0.00033f;
    private bool afterTracking = false;
    private const float targetDiff = 0.95f;

    void Start()
    {
        // アームの長さパラメータの設定
        L1 = 0.1f;
        L2 = 0.305834f;
        L3 = 0.2033f;
        L4 = 0.0967f;
        L5 = 0.07015f;
        L6 = 0.03f;
        if (target == null)
        {
            Debug.LogWarning("ArmController: targetが設定されていません");
        }
        if (endEffector == null)
        {
            Debug.LogWarning("ArmController: endEffectorが設定されていません");
        }
        if (root == null)
        {
            Debug.LogWarning("ArmController: rootが設定されていません");
        }
        if (joints == null || joints.Length != 6)
        {
            Debug.LogWarning("ArmController: joints配列が正しく設定されていません（6つの関節が必要）");
        }
        if (rightFinger == null || leftFinger == null)
        {
            Debug.LogWarning("ArmController: 指のTransformが設定されていません");
        }
        SetInitialPosition();
    }

    void Update()
    {
        if (!enableIK)
        {
            return; // IKが無効化されている場合はreturn
        }
        if (GlobalVariables.controllerState == ControllerState.nochoice || GlobalVariables.resetButtonState == ButtonState.activating)
        {
            return; // コントローラーが選択されていないか、リセット中の場合はreturn
        }
        if (afterTracking)
        {
            Vector3 target_pos = target.position - root.position;
            if (Vector3.Distance(target_pos, endEffector.position - root.position) > 2 * ikRange)
            { // targetとendeffの距離がikRangeの2倍を超えたらトラッキング再開可能
                afterTracking = false;
            }
            else
            {
                return; // まだトラッキング再開可能な距離に達していない場合はreturn
            }
        }
        if ((isLeftHand && !GlobalVariables.isTrackingLeft) || (!isLeftHand && !GlobalVariables.isTrackingRight))
        { // トラッキング中ではない場合
            Vector3 target_pos = target.position - root.position;
            if (Vector3.Distance(target_pos, endEffector.position - root.position) < ikRange)
            { // targetとendeffの距離が指定範囲内に収まっている場合はトラッキング開始
                if (isLeftHand)
                {
                    GlobalVariables.isTrackingLeft = true;
                }
                else
                {
                    GlobalVariables.isTrackingRight = true;
                }
                if (GlobalVariables.isTrackingLeft && GlobalVariables.isTrackingRight)
                {
                    GlobalVariables.isTracking = true;
                }
                GlobalVariables.resetButtonState = ButtonState.deactivated; // トラッキング中はリセットボタン無効化
            }
            else
            {
                return; // 指定範囲外の場合はreturn
            }
        }
        if (OVRInput.GetDown(OVRInput.Button.Two) || OVRInput.GetDown(OVRInput.Button.Four))
        { // B or Yボタンが押されたらトラッキングを停止
            afterTracking = true;
            if (isLeftHand)
            {
                GlobalVariables.isTrackingLeft = false;
            }
            else
            {
                GlobalVariables.isTrackingRight = false;
            }
            // 両手とも停止している場合は全体を停止
            if (!GlobalVariables.isTrackingLeft && !GlobalVariables.isTrackingRight)
            {
                GlobalVariables.isTracking = false;
                GlobalVariables.resetButtonState = ButtonState.active;
            }
            return;
        }
        CalculateInverseKinematics();
        ApplyJointAngles();
    }

    public bool CheckDistance(float range)
    {
        if (target == null || endEffector == null || root == null)
        {
            return false;
        }
        Vector3 target_pos = target.position - root.position;
        if (Vector3.Distance(target_pos, endEffector.position - root.position) > range)
        {
            return false;
        }
        return true;
    }

    private void CalculateInverseKinematics()
    {
        Vector3 target_pos = target.position - root.position;
        float ax, ay, az;
        float asx, asy, asz;
        float p5x, p5y, p5z;
        float C1, C23, S1, S23, C3;
        float px, py, pz, ry, rz;
        // 座標変換
        px = -target_pos.z;
        py = target_pos.x;
        pz = target_pos.y;
        // 回転角度の計算
        Quaternion q = target.rotation * Quaternion.AngleAxis(180, Vector3.up);
        // 特異姿勢回避のためにjoint[2]が向いている方向を取得
        Quaternion joint2Rotation = joints[2].rotation * Quaternion.AngleAxis(90.0f, Vector3.right);
        // 目標回転方向とjoint[2]の方向の内積を計算
        float dotProduct = Quaternion.Dot(q, joint2Rotation);
        float absDot = Mathf.Abs(dotProduct);
        if (isLeftHand)
        {
            Debug.Log("Dot Product: " + dotProduct);
        }
        // dotProductがtargetDiff以上の場合，qの姿勢がjoint2の姿勢から離れるように回転させる
        if (absDot > targetDiff)
        {
            // 回転軸の計算
            Vector3 forwardQ = q * Vector3.forward;
            Vector3 forwardJoint2 = joint2Rotation * Vector3.forward;
            Vector3 rotationAxis = Vector3.Cross(forwardJoint2, forwardQ).normalized;
            // 回転軸が無効な場合のフォールバック
            if (rotationAxis.sqrMagnitude < 0.001f)
            {
                // 代替の回転軸を選択
                rotationAxis = Vector3.Cross(forwardJoint2, Vector3.up).normalized;
                if (rotationAxis.sqrMagnitude < 0.001f)
                {
                    rotationAxis = Vector3.Cross(forwardJoint2, Vector3.right).normalized;
                    if (rotationAxis.sqrMagnitude < 0.001f)
                    {
                        rotationAxis = Vector3.up;
                    }
                }
            }
            // dotProductの絶対値をtargetDiffまで減少させるための回転角度を計算
            float currentAngleDeg = Mathf.Acos(Mathf.Clamp(absDot, 0f, 1f)) * Mathf.Rad2Deg;
            float targetAngleDeg = Mathf.Acos(Mathf.Clamp(targetDiff, 0f, 1f)) * Mathf.Rad2Deg;
            float angleToRotate = targetAngleDeg - currentAngleDeg;
            // dotProductが負の場合は、回転方向を反転
            if (isLeftHand)
            {
                if (dotProduct > 0)
                {
                    angleToRotate = -angleToRotate;
                }
            }
            else
            {
                if (dotProduct < 0)
                {
                    angleToRotate = -angleToRotate;
                }
            }
            Quaternion correctionRotation = Quaternion.AngleAxis(angleToRotate, rotationAxis);
            q = correctionRotation * q;
            if (isLeftHand)
            {
                float newDotProduct = Quaternion.Dot(q, joint2Rotation);
                Debug.Log("Before: " + dotProduct + ", After: " + newDotProduct + ", Angle: " + angleToRotate);
            }
        }
        ry = -q.eulerAngles.x;
        rz = -q.eulerAngles.y;
        // 方向ベクトルの計算
        ax = Mathf.Cos(rz * Mathf.PI / 180.0f) * Mathf.Cos(ry * Mathf.PI / 180.0f);
        ay = Mathf.Sin(rz * Mathf.PI / 180.0f) * Mathf.Cos(ry * Mathf.PI / 180.0f);
        az = -Mathf.Sin(ry * Mathf.PI / 180.0f);
        // 手首位置の計算
        p5x = px - (L5 + L6) * ax;
        p5y = py - (L5 + L6) * ay;
        p5z = pz - (L5 + L6) * az;
        // 第1関節角度
        theta[0] = Mathf.Atan2(p5y, p5x);
        // 第3関節角度
        C3 = (Mathf.Pow(p5x, 2) + Mathf.Pow(p5y, 2) + Mathf.Pow(p5z - L1, 2) - Mathf.Pow(L2, 2) - Mathf.Pow(L3 + L4, 2)) / (2 * L2 * (L3 + L4));
        theta[2] = Mathf.Atan2(Mathf.Pow(1 - Mathf.Pow(C3, 2), 0.5f), C3);
        // 第2関節角度
        float M = L2 + (L3 + L4) * C3;
        float N = (L3 + L4) * Mathf.Sin((float)theta[2]);
        float A = Mathf.Pow(p5x * p5x + p5y * p5y, 0.5f);
        float B = p5z - L1;
        theta[1] = Mathf.Atan2(M * A - N * B, N * A + M * B);
        // 第4、第5関節角度の計算
        C1 = Mathf.Cos((float)theta[0]);
        C23 = Mathf.Cos((float)theta[1] + (float)theta[2]);
        S1 = Mathf.Sin((float)theta[0]);
        S23 = Mathf.Sin((float)theta[1] + (float)theta[2]);
        asx = C23 * (C1 * ax + S1 * ay) - S23 * az;
        asy = -S1 * ax + C1 * ay;
        asz = S23 * (C1 * ax + S1 * ay) + C23 * az;
        theta[3] = Mathf.Atan2(asy, asx);
        theta[4] = Mathf.Atan2(Mathf.Cos((float)theta[3]) * asx + Mathf.Sin((float)theta[3]) * asy, asz);
        // 符号反転処理
        theta[0] = -theta[0];
        theta[1] = -theta[1] + 0.19599f;
        theta[2] = -theta[2] + Mathf.PI / 2.0f - 0.19599f;
        theta[3] = -theta[3];
        theta[4] = -theta[4];
        Vector3 joint4Up = joints[4].transform.right;
        Vector3 targetUp = target.up;
        Vector3 rollAxis = -target.forward;
        float rollAngleDegrees = Vector3.SignedAngle(joint4Up, targetUp, rollAxis);
        theta[5] = Mathf.PI / 2 - rollAngleDegrees * Mathf.Deg2Rad;
    }

    private void ApplyJointAngles()
    {
        if (joints == null || joints.Length != 6)
        {
            return;
        }
        // NaNチェックと関節角度の適用
        if (!double.IsNaN(theta[0]))
            joints[0].transform.localEulerAngles = new Vector3(0, 0, (float)theta[0] * Mathf.Rad2Deg);
        if (!double.IsNaN(theta[1]))
            joints[1].transform.localEulerAngles = new Vector3((float)theta[1] * Mathf.Rad2Deg, 0, 0);
        if (!double.IsNaN(theta[2]))
            joints[2].transform.localEulerAngles = new Vector3((float)theta[2] * Mathf.Rad2Deg, 0, 0);
        if (!double.IsNaN(theta[3]))
            joints[3].transform.localEulerAngles = new Vector3(0, (float)theta[3] * Mathf.Rad2Deg, 0);
        if (!double.IsNaN(theta[4]))
            joints[4].transform.localEulerAngles = new Vector3((float)theta[4] * Mathf.Rad2Deg, 0, 0);
        if (!double.IsNaN(theta[5]))
            joints[5].transform.localEulerAngles = new Vector3(0, (float)theta[5] * Mathf.Rad2Deg, 0);
        if (isLeftHand)
        {
            triggerPower = OVRInput.Get(OVRInput.Axis1D.PrimaryHandTrigger, OVRInput.Controller.LTouch);
        }
        else
        {
            triggerPower = OVRInput.Get(OVRInput.Axis1D.PrimaryHandTrigger, OVRInput.Controller.RTouch);
        }
        triggerPower = Mathf.Pow(triggerPower, 0.3f);
        rightFinger.transform.localPosition = new Vector3(triggerPower * fingerPower, rightFinger.transform.localPosition.y, rightFinger.transform.localPosition.z);
        leftFinger.transform.localPosition = new Vector3(triggerPower * -fingerPower, leftFinger.transform.localPosition.y, leftFinger.transform.localPosition.z);
    }

    public void SetIKEnabled(bool enabled)
    {
        enableIK = enabled;
    }

    public float[] GetJointAnglesDegrees()
    {
        float[] angles = new float[7];
        for (int i = 0; i < 6; i++)
        {
            angles[i] = (float)theta[i];
        }
        angles[6] = triggerPower;
        return angles;
    }

    public bool IsLeftHand => isLeftHand;

    public void SetInitialPosition()
    {
        if (isLeftHand)
        {
            theta[0] = -Math.PI / 2.0;
        }
        else
        {
            theta[0] = Math.PI / 2.0;
        }
        theta[1] = Math.PI / 2.0;
        theta[2] = -85.0 * Mathf.Deg2Rad;
        theta[3] = 0.0;
        theta[4] = 0.0;
        theta[5] = 0.0;
        ApplyJointAngles();
    }
}
